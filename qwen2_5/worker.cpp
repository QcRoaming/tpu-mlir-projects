// BMRuntime worker: KV stays on device; decoding copies only one new KV row.
#include "bmlib_runtime.h"
#include "bmruntime_interface.h"

#include <algorithm>
#include <chrono>
#include <climits>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using Clock = std::chrono::steady_clock;
static double elapsed(Clock::time_point t) {
  return std::chrono::duration<double, std::milli>(Clock::now() - t).count();
}
static void require(bool condition, const std::string &message) {
  if (!condition)
    throw std::runtime_error(message);
}
static void check(bm_status_t status, const char *operation) {
  require(status == BM_SUCCESS,
          std::string(operation) + " failed: " + std::to_string(status));
}
static std::string escape(const std::string &s) {
  std::string out;
  for (unsigned char c : s) {
    if (c == '\\' || c == '"')
      out += '\\';
    if (c >= 32)
      out += c;
  }
  return out;
}
static void reply(const std::string &json) {
  // SDK diagnostics may also use stdout; frontend accepts only this prefix.
  std::cout << "QWEN_JSON " << json << std::endl;
}

struct Net {
  const bm_net_info_t *info;
  std::string name;
};
struct Timing {
  double prepare = 0, embedding = 0, blocks = 0, head = 0, cache_update = 0,
         total = 0;
};

class Engine {
public:
  Engine(const std::string &model, int device) {
    try {
      check(bm_dev_request(&handle_, device), "bm_dev_request");
      unsigned int chip = 0;
      check(bm_get_chipid(handle_, &chip), "bm_get_chipid");
      require(chip == 0x1686,
              "This sample requires one BM1684X device (SC7 HP75)");
      runtime_ = bmrt_create(handle_);
      require(runtime_, "bmrt_create failed");
      require(bmrt_load_bmodel(runtime_, model.c_str()),
              "Cannot load local bmodel");
      embedding_ = net("embedding", 1, 1);
      embedding_cache_ = net("embedding_cache", 1, 1);
      head_ = net("lm_head", 1, 1);
      const auto &shape = embedding_.info->stages[0].input_shapes[0];
      require(shape.num_dims == 2 && shape.dims[0] == 1 && shape.dims[1] >= 2,
              "embedding must have static shape [1, seq_length]");
      seq_ = shape.dims[1];
      const auto &hidden = embedding_.info->stages[0].output_shapes[0];
      require(hidden.num_dims == 3 && hidden.dims[0] == 1 &&
                  hidden.dims[1] == seq_,
              "embedding output must be [1, seq_length, hidden_size]");
      hidden_size_ = hidden.dims[2];
      require(embedding_.info->input_dtypes[0] == BM_INT32 &&
                  embedding_cache_.info->input_dtypes[0] == BM_INT32,
              "embedding input dtype must be INT32");
      require(bmrt_shape_count(&head_.info->stages[0].output_shapes[0]) == 1 &&
                  head_.info->output_dtypes[0] == BM_INT32,
              "lm_head must include greedy TopK (compile without --do_sample)");
      for (int i = 0; bmrt_get_network_info(
               runtime_, ("block_" + std::to_string(i)).c_str());
           ++i) {
        prefill_.push_back(net("block_" + std::to_string(i), 3, 3));
        decode_.push_back(net("block_cache_" + std::to_string(i), 5, 3));
      }
      require(!prefill_.empty(), "No block_<layer> networks");
      require(bmrt_get_network_number(runtime_) ==
                  3 + 2 * static_cast<int>(prefill_.size()),
              "Unexpected networks: use single-device batch=1 without "
              "dynamic/history/chunk/sample/LoRA");
      validate_layout();
      input_prefill_ = alloc(input_bytes(embedding_, 0));
      input_decode_ = alloc(input_bytes(embedding_cache_, 0));
      position_prefill_ = alloc(input_bytes(prefill_[0], 1));
      position_decode_ = alloc(input_bytes(decode_[0], 1));
      mask_prefill_ = alloc(input_bytes(prefill_[0], 2));
      mask_decode_ = alloc(input_bytes(decode_[0], 2));
      hidden_a_ = alloc(output_bytes(embedding_, 0));
      hidden_b_ = alloc(output_bytes(embedding_, 0));
      head_output_ = alloc(output_bytes(head_, 0));
      scratch_k_ = alloc(output_bytes(decode_[0], 1));
      scratch_v_ = alloc(output_bytes(decode_[0], 2));
      kv_row_bytes_ = output_bytes(decode_[0], 1);
      for (size_t i = 0; i < prefill_.size(); ++i) {
        cache_k_.push_back(alloc(input_bytes(decode_[i], 3)));
        cache_v_.push_back(alloc(input_bytes(decode_[i], 4)));
        kv_bytes_ += input_bytes(decode_[i], 3) + input_bytes(decode_[i], 4);
      }
      initialize_prefill_mask();
    } catch (...) {
      cleanup();
      throw;
    }
  }
  Engine(const Engine &) = delete;
  Engine &operator=(const Engine &) = delete;
  ~Engine() { cleanup(); }

  std::string info(double load_ms) const {
    std::ostringstream s;
    s << std::setprecision(12) << "{\"event\":\"ready\",\"seq_length\":" << seq_
      << ",\"supports_append\":true"
      << ",\"layers\":" << prefill_.size()
      << ",\"hidden_size\":" << hidden_size_ << ",\"kv_bytes\":" << kv_bytes_
      << ",\"owned_device_bytes\":" << owned_bytes_
      << ",\"workspace_bytes\":" << owned_bytes_ - kv_bytes_
      << ",\"model_load_ms\":" << load_ms
      << ",\"memory_scope\":\"sample-owned device buffers; excludes BMRuntime "
         "weights and internal workspaces\"}";
    return s.str();
  }
  void reset() {
    invalidate();
    for (size_t i = 0; i < cache_k_.size(); ++i) {
      check(bm_memset_device(handle_, 0, cache_k_[i]), "clear K cache");
      check(bm_memset_device(handle_, 0, cache_v_[i]), "clear V cache");
    }
    check(bm_thread_sync(handle_), "reset sync");
  }
  void invalidate() {
    length_ = 0;
    next_token_ready_ = false;
  }

  std::string prefill(const std::vector<int32_t> &tokens) {
    const auto start = Clock::now();
    Timing t;
    invalidate(); // An interrupted prefill must not leave a usable old session.
    require(!tokens.empty() && tokens.size() < static_cast<size_t>(seq_),
            "Prompt length must be between 1 and seq_length-1");
    auto stage = Clock::now();
    std::vector<int32_t> ids(seq_, 0), positions(seq_, 0);
    std::copy(tokens.begin(), tokens.end(), ids.begin());
    for (size_t i = 0; i < tokens.size(); ++i)
      positions[i] = static_cast<int32_t>(i);
    upload(input_prefill_, ids.data(), ids.size() * sizeof(int32_t));
    upload(position_prefill_, positions.data(),
           positions.size() * sizeof(int32_t));
    t.prepare = elapsed(stage);
    stage = Clock::now();
    launch(embedding_, {input_prefill_}, {hidden_a_});
    t.embedding = elapsed(stage);
    auto current = hidden_a_, next = hidden_b_;
    stage = Clock::now();
    for (size_t i = 0; i < prefill_.size(); ++i) {
      launch(prefill_[i], {current, position_prefill_, mask_prefill_},
             {next, cache_k_[i], cache_v_[i]});
      std::swap(current, next);
    }
    t.blocks = elapsed(stage);
    length_ = static_cast<int>(tokens.size());
    stage = Clock::now();
    const size_t hidden_row = input_bytes(head_, 0);
    last_token_ =
        predict(slice(current, (length_ - 1) * hidden_row, hidden_row));
    next_token_ready_ = true;
    t.head = elapsed(stage);
    t.total = elapsed(start);
    return token_reply("prefill", t);
  }

  std::string decode() {
    require(next_token_ready_,
            "No predicted token is ready; run PREFILL/APPEND");
    return token_reply("decode", step(last_token_, true));
  }

  std::string append(int expected_length, const std::vector<int32_t> &tokens) {
    const auto start = Clock::now();
    Timing t;
    require(length_ > 0 && expected_length == length_,
            "APPEND cache length mismatch; run PREFILL/RESET");
    require(!tokens.empty() && length_ < seq_ &&
                tokens.size() < static_cast<size_t>(seq_ - length_),
            "APPEND requires tokens and one remaining output position");
    for (size_t i = 0; i < tokens.size(); ++i) {
      // The supplied prompt suffix, not a sampled prediction, advances KV.
      const Timing current = step(tokens[i], i + 1 == tokens.size());
      t.prepare += current.prepare;
      t.embedding += current.embedding;
      t.blocks += current.blocks;
      t.head += current.head;
      t.cache_update += current.cache_update;
    }
    t.total = elapsed(start);
    return token_reply("append", t);
  }

private:
  bm_handle_t handle_ = nullptr;
  void *runtime_ = nullptr;
  Net embedding_{}, embedding_cache_{}, head_{};
  std::vector<Net> prefill_, decode_;
  std::vector<bm_device_mem_t> owned_, cache_k_, cache_v_;
  bm_device_mem_t input_prefill_{}, input_decode_{}, position_prefill_{},
      position_decode_{};
  bm_device_mem_t mask_prefill_{}, mask_decode_{}, hidden_a_{}, hidden_b_{},
      head_output_{};
  bm_device_mem_t scratch_k_{}, scratch_v_{};
  int seq_ = 0, hidden_size_ = 0, length_ = 0;
  int32_t last_token_ = 0;
  bool next_token_ready_ = false;
  size_t owned_bytes_ = 0, kv_bytes_ = 0, kv_row_bytes_ = 0;

  Timing step(int32_t token, bool predict_next) {
    const auto start = Clock::now();
    Timing t;
    require(length_ > 0 && length_ < seq_,
            "No active session or context is full; run PREFILL/RESET");
    require(token >= 0, "Invalid token ID");
    // Intermediate APPEND steps leave no prediction that DECODE may consume.
    next_token_ready_ = false;
    auto stage = Clock::now();
    int32_t position = length_;
    upload(input_decode_, &token, sizeof(token));
    upload(position_decode_, &position, sizeof(position));
    auto mask = attention_mask(decode_[0].info->input_dtypes[2], seq_ + 1);
    for (int i = 0; i < length_; ++i)
      unmask(mask, decode_[0].info->input_dtypes[2], i);
    // Current K/V is appended after the entire S-slot history, NOT at length_.
    unmask(mask, decode_[0].info->input_dtypes[2], seq_);
    upload(mask_decode_, mask.data(), mask.size());
    t.prepare = elapsed(stage);
    stage = Clock::now();
    launch(embedding_cache_, {input_decode_}, {hidden_a_});
    t.embedding = elapsed(stage);
    auto current = hidden_a_, next = hidden_b_;
    for (size_t i = 0; i < decode_.size(); ++i) {
      stage = Clock::now();
      launch(
          decode_[i],
          {current, position_decode_, mask_decode_, cache_k_[i], cache_v_[i]},
          {next, scratch_k_, scratch_v_});
      t.blocks += elapsed(stage);
      stage = Clock::now();
      check(bm_memcpy_d2d_byte(handle_, cache_k_[i], length_ * kv_row_bytes_,
                               scratch_k_, 0, kv_row_bytes_),
            "append K row");
      check(bm_memcpy_d2d_byte(handle_, cache_v_[i], length_ * kv_row_bytes_,
                               scratch_v_, 0, kv_row_bytes_),
            "append V row");
      check(bm_thread_sync(handle_), "KV append sync");
      t.cache_update += elapsed(stage);
      std::swap(current, next);
    }
    ++length_;
    if (predict_next) {
      stage = Clock::now();
      last_token_ = predict(current);
      next_token_ready_ = true;
      t.head = elapsed(stage);
    }
    t.total = elapsed(start);
    return t;
  }

  void initialize_prefill_mask() {
    const auto dtype = prefill_[0].info->input_dtypes[2];
    auto mask = attention_mask(dtype, seq_ * size_t(seq_));
    for (int row = 0; row < seq_; ++row) {
      // This causal mask is constant across prompts, including padded rows.
      for (int col = 0; col <= row; ++col)
        unmask(mask, dtype, row * size_t(seq_) + col);
    }
    upload(mask_prefill_, mask.data(), mask.size());
    check(bm_thread_sync(handle_), "prefill mask initialization sync");
  }

  void cleanup() {
    if (handle_)
      for (auto m : owned_)
        bm_free_device(handle_, m);
    owned_.clear();
    if (runtime_)
      bmrt_destroy(runtime_);
    runtime_ = nullptr;
    if (handle_)
      bm_dev_free(handle_);
    handle_ = nullptr;
  }
  Net net(const std::string &name, int inputs, int outputs) {
    auto *p = bmrt_get_network_info(runtime_, name.c_str());
    require(p, "Missing network: " + name);
    require(!p->is_dynamic && p->stage_num == 1 && p->input_num == inputs &&
                p->output_num == outputs,
            "Unsupported network signature: " + name);
    for (int i = 0; i < inputs; ++i)
      require(p->input_loc_devices[i] == 0,
              "Multi-device input is unsupported");
    for (int i = 0; i < outputs; ++i)
      require(p->output_loc_devices[i] == 0,
              "Multi-device output is unsupported");
    return {p, name};
  }
  static size_t bytes(bm_shape_t shape, bm_data_type_t dtype) {
    return bmrt_shape_count(&shape) * bmrt_data_type_size(dtype);
  }
  static size_t input_bytes(const Net &n, int i) {
    return bytes(n.info->stages[0].input_shapes[i], n.info->input_dtypes[i]);
  }
  static size_t output_bytes(const Net &n, int i) {
    return bytes(n.info->stages[0].output_shapes[i], n.info->output_dtypes[i]);
  }
  static bool shape_is(const bm_shape_t &s, std::initializer_list<int> dims) {
    return s.num_dims == static_cast<int>(dims.size()) &&
           std::equal(dims.begin(), dims.end(), s.dims);
  }
  static bool floating(bm_data_type_t t) {
    return t == BM_BFLOAT16 || t == BM_FLOAT16 || t == BM_FLOAT32;
  }
  void validate_layout() {
    const auto activation = embedding_.info->output_dtypes[0];
    require(floating(activation), "Unsupported activation dtype");
    require(
        shape_is(embedding_cache_.info->stages[0].input_shapes[0], {1, 1}) &&
            shape_is(embedding_cache_.info->stages[0].output_shapes[0],
                     {1, 1, hidden_size_}) &&
            embedding_cache_.info->output_dtypes[0] == activation &&
            shape_is(head_.info->stages[0].input_shapes[0],
                     {1, hidden_size_}) &&
            head_.info->input_dtypes[0] == activation,
        "Embedding/head shape or dtype mismatch");
    const auto cache_shape = decode_[0].info->stages[0].input_shapes[3];
    require(cache_shape.num_dims == 4 && cache_shape.dims[0] == 1 &&
                cache_shape.dims[1] == seq_,
            "KV must have shape [1, seq_length, kv_heads, head_dim]");
    for (size_t i = 0; i < prefill_.size(); ++i) {
      const auto *p = prefill_[i].info, *d = decode_[i].info;
      require(
          shape_is(p->stages[0].input_shapes[0], {1, seq_, hidden_size_}) &&
              shape_is(p->stages[0].output_shapes[0],
                       {1, seq_, hidden_size_}) &&
              shape_is(p->stages[0].input_shapes[1], {1, seq_}) &&
              shape_is(p->stages[0].input_shapes[2], {1, 1, seq_, seq_}) &&
              shape_is(d->stages[0].input_shapes[0], {1, 1, hidden_size_}) &&
              shape_is(d->stages[0].output_shapes[0], {1, 1, hidden_size_}) &&
              shape_is(d->stages[0].input_shapes[1], {1, 1}) &&
              shape_is(d->stages[0].input_shapes[2], {1, 1, 1, seq_ + 1}),
          "Unexpected attention layout at layer " + std::to_string(i));
      require(p->input_dtypes[0] == activation &&
                  p->output_dtypes[0] == activation &&
                  d->input_dtypes[0] == activation &&
                  d->output_dtypes[0] == activation &&
                  p->input_dtypes[1] == BM_INT32 &&
                  d->input_dtypes[1] == BM_INT32 &&
                  floating(p->input_dtypes[2]) &&
                  floating(d->input_dtypes[2]) &&
                  p->input_dtypes[2] == prefill_[0].info->input_dtypes[2] &&
                  d->input_dtypes[2] == decode_[0].info->input_dtypes[2],
              "Attention dtype mismatch");
      for (int k = 0; k < 2; ++k) {
        require(
            bmrt_shape_is_same(&p->stages[0].output_shapes[k + 1],
                               &cache_shape) &&
                bmrt_shape_is_same(&d->stages[0].input_shapes[k + 3],
                                   &cache_shape) &&
                shape_is(d->stages[0].output_shapes[k + 1],
                         {1, 1, cache_shape.dims[2], cache_shape.dims[3]}) &&
                p->output_dtypes[k + 1] == d->input_dtypes[k + 3] &&
                d->output_dtypes[k + 1] == d->input_dtypes[k + 3] &&
                d->input_dtypes[k + 3] == decode_[0].info->input_dtypes[3],
            "KV layout/dtype mismatch");
      }
    }
  }
  bm_device_mem_t alloc(size_t size) {
    require(size > 0 && size <= UINT_MAX, "Unsupported device allocation size");
    bm_device_mem_t mem;
    check(bm_malloc_device_byte(handle_, &mem, static_cast<unsigned int>(size)),
          "bm_malloc_device_byte");
    owned_.push_back(mem);
    owned_bytes_ += size;
    return mem;
  }
  static bm_device_mem_t slice(bm_device_mem_t mem, size_t offset,
                               size_t size) {
    require(offset + size <= bm_mem_get_device_size(mem) && size <= UINT_MAX,
            "Device slice out of range");
    return bm_mem_from_device(bm_mem_get_device_addr(mem) + offset,
                              static_cast<unsigned int>(size));
  }
  void upload(bm_device_mem_t mem, const void *data, size_t size) {
    require(size <= UINT_MAX && size <= bm_mem_get_device_size(mem),
            "Host input exceeds device buffer");
    check(bm_memcpy_s2d_partial(handle_, mem, const_cast<void *>(data),
                                static_cast<unsigned int>(size)),
          "upload input");
  }
  void launch(const Net &net, const std::vector<bm_device_mem_t> &input,
              const std::vector<bm_device_mem_t> &output) {
    std::vector<bm_tensor_t> inputs(input.size()), outputs(output.size());
    for (size_t i = 0; i < input.size(); ++i) {
      bmrt_tensor_with_device(
          &inputs[i], slice(input[i], 0, input_bytes(net, static_cast<int>(i))),
          net.info->input_dtypes[i], net.info->stages[0].input_shapes[i]);
    }
    for (size_t i = 0; i < output.size(); ++i) {
      bmrt_tensor_with_device(
          &outputs[i],
          slice(output[i], 0, output_bytes(net, static_cast<int>(i))),
          net.info->output_dtypes[i], net.info->stages[0].output_shapes[i]);
    }
    require(
        bmrt_launch_tensor_ex(runtime_, net.name.c_str(), inputs.data(),
                              static_cast<int>(inputs.size()), outputs.data(),
                              static_cast<int>(outputs.size()), true, false),
        "Network launch failed: " + net.name);
    check(bm_thread_sync(handle_), "network sync");
  }
  static std::vector<uint8_t> attention_mask(bm_data_type_t dtype,
                                             size_t count) {
    const size_t width = bmrt_data_type_size(dtype);
    std::vector<uint8_t> result(count * width);
    if (dtype == BM_FLOAT32) {
      const float value = -10000.f;
      for (size_t i = 0; i < count; ++i)
        std::memcpy(result.data() + i * 4, &value, 4);
    } else {
      require(dtype == BM_BFLOAT16 || dtype == BM_FLOAT16,
              "Unsupported mask dtype");
      const uint16_t value = dtype == BM_BFLOAT16 ? 0xc61c : 0xf0e2;
      for (size_t i = 0; i < count; ++i)
        std::memcpy(result.data() + i * 2, &value, 2);
    }
    return result;
  }
  static void unmask(std::vector<uint8_t> &mask, bm_data_type_t dtype,
                     size_t index) {
    const size_t width = bmrt_data_type_size(dtype);
    std::memset(mask.data() + index * width, 0, width);
  }
  int32_t predict(bm_device_mem_t hidden) {
    launch(head_, {hidden}, {head_output_});
    int32_t token = 0;
    check(bm_memcpy_d2s_partial(handle_, &token, head_output_, sizeof(token)),
          "read predicted token");
    require(token >= 0, "Invalid predicted token ID");
    return token;
  }
  std::string token_reply(const std::string &phase, const Timing &t) const {
    std::ostringstream s;
    s << std::setprecision(12) << "{\"event\":\"token\",\"phase\":\"" << phase
      << "\",\"token_id\":" << last_token_ << ",\"cached_tokens\":" << length_
      << ",\"prepare_ms\":" << t.prepare << ",\"embedding_ms\":" << t.embedding
      << ",\"blocks_ms\":" << t.blocks << ",\"head_ms\":" << t.head
      << ",\"cache_update_ms\":" << t.cache_update
      << ",\"total_ms\":" << t.total << '}';
    return s.str();
  }
};

int main(int argc, char **argv) {
  if (argc < 2 || argc > 3) {
    std::cerr << "Usage: qwen_bmrt LOCAL.bmodel [DEVICE_ID]\n";
    return 2;
  }
  try {
    const auto load = Clock::now();
    const int device = argc == 3 ? std::stoi(argv[2]) : 0;
    require(device >= 0, "DEVICE_ID must be nonnegative");
    Engine engine(argv[1], device);
    reply(engine.info(elapsed(load)));
    std::string line;
    while (std::getline(std::cin, line)) {
      try {
        std::istringstream in(line);
        std::string command;
        in >> command;
        if (command == "QUIT")
          break;
        if (command == "RESET") {
          engine.reset();
          reply("{\"event\":\"reset\"}");
        } else if (command == "DECODE") {
          reply(engine.decode());
        } else if (command == "PREFILL" || command == "APPEND") {
          int expected_length = 0;
          if (command == "APPEND")
            require(static_cast<bool>(in >> expected_length),
                    "Invalid expected cache length");
          int n = 0;
          require(static_cast<bool>(in >> n) && n > 0 && n <= 65536,
                  "Invalid prompt token count");
          std::vector<int32_t> tokens(static_cast<size_t>(n));
          for (auto &token : tokens) {
            require(static_cast<bool>(in >> token) && token >= 0,
                    "Invalid or missing token ID");
          }
          std::string extra;
          require(!(in >> extra), "Unexpected data after prompt IDs");
          reply(command == "APPEND" ? engine.append(expected_length, tokens)
                                    : engine.prefill(tokens));
        } else {
          throw std::runtime_error("Unknown command");
        }
      } catch (const std::exception &e) {
        engine.invalidate();
        reply("{\"event\":\"error\",\"error\":\"" + escape(e.what()) + "\"}");
      }
    }
  } catch (const std::exception &e) {
    reply("{\"event\":\"error\",\"error\":\"" + escape(e.what()) + "\"}");
    return 1;
  }
  return 0;
}
