#include <functional>
#include <vector>

// this needs to be rewritten in CUDA
struct Embedder 
{
	std::vector<std::function<float(float)>> embed_fns;
	
	Embedder() {
		int max_freq = 9;
		int n_freqs = 10;
		
		std::vector<float> freq_bands;
		for (int i = 0; i < n_freqs; i++) {
			freq_bands.push_back(powf(2.0f, (float)i));
		}
		
		
		std::vector<std::function<float(float)>> periodic_fns = {
			[](float x) { return sinf(x); },
			[](float x) { return cosf(x); }
		};
		
		for (const auto& freq : freq_bands) {
			for (const auto p_fn : periodic_fns) {
				embed_fns.emplace_back([=](float x) { return p_fn(x * freq); });
			}
		}
		
	}
	void embed(float* values) {
		for (int i = 0; i < embed_fns.size(); i++) {
			values[i] = embed_fns[i](values[i]);
		}
	}
};