// Copyright 2013, Takuya Akiba
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
//     * Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//     * Redistributions in binary form must reproduce the above
// copyright notice, this list of conditions and the following disclaimer
// in the documentation and/or other materials provided with the
// distribution.
//     * Neither the name of Takuya Akiba nor the names of its
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#ifndef PRUNED_LANDMARK_LABELING_H_
#define PRUNED_LANDMARK_LABELING_H_

#include <malloc.h>
#include <stdint.h>
#include <xmmintrin.h>
#include <sys/time.h>
#include <climits>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <stack>
#include <queue>
#include <set>
#include <algorithm>
#include <fstream>
#include <utility>

// Add by Johnpzh
#include "/home/zpeng/pppp/pado/includes/globals.h"
// End by Johnpzh

//
// NOTE: Currently only unweighted and undirected graphs are supported.
//

template<int kNumBitParallelRoots = 50>
class PrunedLandmarkLabeling {
 public:
  // Constructs an index from a graph, given as a list of edges.
  // Vertices should be described by numbers starting from zero.
  // Returns |true| when successful.
  bool ConstructIndex(const std::vector<std::pair<int, int> > &es);
  bool ConstructIndex(std::istream &ifs);
  bool ConstructIndex(const char *filename);

  // Returns distance vetween vertices |v| and |w| if they are connected.
  // Otherwise, returns |INT_MAX|.
  inline int QueryDistance(int v, int w);

  // Loads an index. Returns |true| when successful.
  bool LoadIndex(std::istream &ifs);
  bool LoadIndex(const char *filename);

  // Stores the index. Returns |true| when successful.
  bool StoreIndex(std::ostream &ofs);
  bool StoreIndex(const char *filename);

  int GetNumVertices() { return num_v_; }
  void Free();
  void PrintStatistics();

  PrunedLandmarkLabeling()
      : num_v_(0), index_(NULL), time_load_(0), time_indexing_(0) {}
  virtual ~PrunedLandmarkLabeling() {
    Free();
  }

 private:
  static const uint8_t INF8;  // For unreachable pairs

  struct index_t {
//    uint8_t bpspt_d[kNumBitParallelRoots];
//    uint64_t bpspt_s[kNumBitParallelRoots][2];  // [0]: S^{-1}, [1]: S^{0}
    uint32_t *spt_v;
    uint8_t *spt_d;
  } __attribute__((aligned(64)));  // Aligned for cache lines

  int num_v_;
  index_t *index_;

  double GetCurrentTimeSec() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
  }

  // Statistics
  double time_load_, time_indexing_;
  // By Johnpzh
  uint64_t total_check_count = 0;
  uint64_t bp_hit_count = 0;
  uint64_t normal_check_count = 0;
//  PADO::L2CacheMissRate cache_misses;
//  PADO::TotalInstructsExe bp_labeling_ins_count;
//  PADO::TotalInstructsExe bp_checking_ins_count;
//  PADO::TotalInstructsExe initializing_ins_count;
//  PADO::TotalInstructsExe adding_ins_count;
  PADO::TotalInstructsExe dist_query_ins_count;
//  double bp_labeling = 0;
//  double preparing_time = 0;
//  double initialzing_time = 0;
//  double adding_time = 0;
//  double updating_time = 0;
  double distance_query_time = 0;
  // End By Johnpzh
};

template<int kNumBitParallelRoots>
const uint8_t PrunedLandmarkLabeling<kNumBitParallelRoots>::INF8 = 100;

template<int kNumBitParallelRoots>
bool PrunedLandmarkLabeling<kNumBitParallelRoots>
::ConstructIndex(const char *filename) {
  std::ifstream ifs(filename);
  return ifs && ConstructIndex(ifs);
}

template<int kNumBitParallelRoots>
bool PrunedLandmarkLabeling<kNumBitParallelRoots>
::ConstructIndex(std::istream &ifs) {
  std::vector<std::pair<int, int> > es;
  for (int v, w; ifs >> v >> w; ) {
    es.push_back(std::make_pair(v, w));
  }
  if (ifs.bad()) return false;
  ConstructIndex(es);
  return true;
}

template<int kNumBitParallelRoots>
bool PrunedLandmarkLabeling<kNumBitParallelRoots>
::ConstructIndex(const std::vector<std::pair<int, int> > &es) {
  //
  // Prepare the adjacency list and index space
  //
  Free();
  time_load_ = -GetCurrentTimeSec();
  int E = es.size();
  int &V = num_v_;
  V = 0;
  for (size_t i = 0; i < es.size(); ++i) {
    V = std::max(V, std::max(es[i].first, es[i].second) + 1);
  }
  std::vector<std::vector<int> > adj(V);
  for (size_t i = 0; i < es.size(); ++i) {
    int v = es[i].first, w = es[i].second;
    adj[v].push_back(w);
    adj[w].push_back(v);
  }
  time_load_ += GetCurrentTimeSec();

  index_ = (index_t*)memalign(64, V * sizeof(index_t));
  if (index_ == NULL) {
    num_v_ = 0;
    return false;
  }
  for (int v = 0; v < V; ++v) {
    index_[v].spt_v = NULL;
    index_[v].spt_d = NULL;
  }

  //
  // Order vertices by decreasing order of degree
  //
  time_indexing_ = -GetCurrentTimeSec();
// Add by Johnpzh
//	cache_misses.measure_start();
//	preparing_time -= GetCurrentTimeSec();
// End by Johnpzh
  std::vector<int> inv(V);  // new label -> old label
  {
    // Order
    std::vector<std::pair<float, int> > deg(V);
    for (int v = 0; v < V; ++v) {
      // We add a random value here to diffuse nearby vertices
      deg[v] = std::make_pair(adj[v].size() + float(rand()) / RAND_MAX, v);
    }
    std::sort(deg.rbegin(), deg.rend());
    for (int i = 0; i < V; ++i) inv[i] = deg[i].second;

    // Relabel the vertex IDs
    std::vector<int> rank(V);
    for (int i = 0; i < V; ++i) rank[deg[i].second] = i;
    std::vector<std::vector<int> > new_adj(V);
    for (int v = 0; v < V; ++v) {
      for (size_t i = 0; i < adj[v].size(); ++i) {
        new_adj[rank[v]].push_back(rank[adj[v][i]]);
      }
    }
    adj.swap(new_adj);
  }

  // Add by Johnpzh
//  preparing_time += GetCurrentTimeSec();
//  bp_labeling -= GetCurrentTimeSec();
//  bp_labeling_ins_count.measure_start();
  // End by Johnpzh

  //
  // Bit-parallel labeling
  //
  std::vector<bool> usd(V, false);  // Used as root? (in new label)
//  {
//    std::vector<uint8_t> tmp_d(V);
//    std::vector<std::pair<uint64_t, uint64_t> > tmp_s(V);
//    std::vector<int> que(V);
//    std::vector<std::pair<int, int> > sibling_es(E);
//    std::vector<std::pair<int, int> > child_es(E);
//
//    int r = 0;
//    for (int i_bpspt = 0; i_bpspt < kNumBitParallelRoots; ++i_bpspt) {
//      while (r < V && usd[r]) ++r;
//      if (r == V) {
//        for (int v = 0; v < V; ++v) index_[v].bpspt_d[i_bpspt] = INF8;
//        continue;
//      }
//      usd[r] = true;
//
//      fill(tmp_d.begin(), tmp_d.end(), INF8);
//      fill(tmp_s.begin(), tmp_s.end(), std::make_pair(0, 0));
//
//      int que_t0 = 0, que_t1 = 0, que_h = 0;
//      que[que_h++] = r;
//      tmp_d[r] = 0;
//      que_t1 = que_h;
//
//      int ns = 0;
//      std::vector<int> vs;
//      sort(adj[r].begin(), adj[r].end());
//      for (size_t i = 0; i < adj[r].size(); ++i) {
//        int v = adj[r][i];
//        if (!usd[v]) {
//          usd[v] = true;
//          que[que_h++] = v;
//          tmp_d[v] = 1;
//          tmp_s[v].first = 1ULL << ns;
//          vs.push_back(v);
//          if (++ns == 64) break;
//        }
//      }
//
//      for (int d = 0; que_t0 < que_h; ++d) {
//        int num_sibling_es = 0, num_child_es = 0;
//
//        for (int que_i = que_t0; que_i < que_t1; ++que_i) {
//          int v = que[que_i];
//
//          for (size_t i = 0; i < adj[v].size(); ++i) {
//            int tv = adj[v][i];
//            int td = d + 1;
//
//            if (d > tmp_d[tv]);
//            else if (d == tmp_d[tv]) {
//              if (v < tv) {
//                sibling_es[num_sibling_es].first  = v;
//                sibling_es[num_sibling_es].second = tv;
//                ++num_sibling_es;
//              }
//            } else {
//              if (tmp_d[tv] == INF8) {
//                que[que_h++] = tv;
//                tmp_d[tv] = td;
//              }
//              child_es[num_child_es].first  = v;
//              child_es[num_child_es].second = tv;
//              ++num_child_es;
//            }
//          }
//        }
//
//        for (int i = 0; i < num_sibling_es; ++i) {
//          int v = sibling_es[i].first, w = sibling_es[i].second;
//          tmp_s[v].second |= tmp_s[w].first;
//          tmp_s[w].second |= tmp_s[v].first;
//        }
//        for (int i = 0; i < num_child_es; ++i) {
//          int v = child_es[i].first, c = child_es[i].second;
//          tmp_s[c].first  |= tmp_s[v].first;
//          tmp_s[c].second |= tmp_s[v].second;
//        }
//
//        que_t0 = que_t1;
//        que_t1 = que_h;
//      }
//
//      for (int v = 0; v < V; ++v) {
//        index_[inv[v]].bpspt_d[i_bpspt] = tmp_d[v];
//        index_[inv[v]].bpspt_s[i_bpspt][0] = tmp_s[v].first;
//        index_[inv[v]].bpspt_s[i_bpspt][1] = tmp_s[v].second & ~tmp_s[v].first;
//      }
//    }
//  }

  // Add by Johnpzh
//  bp_labeling += GetCurrentTimeSec();
//  bp_labeling_ins_count.measure_stop();
  // End by Johnpzh

  //
  // Pruned labeling
  //
  {
	  // Add by Johnpzh
//	  initialzing_time -= GetCurrentTimeSec();
	  // End by Johnpzh
    // Sentinel (V, INF8) is added to all the vertices
    std::vector<std::pair<std::vector<int>, std::vector<uint8_t> > >
        tmp_idx(V, make_pair(std::vector<int>(1, V),
                             std::vector<uint8_t>(1, INF8)));

    std::vector<bool> vis(V);
    std::vector<int> que(V);
    std::vector<uint8_t> dst_r(V + 1, INF8);
	  // Add by Johnpzh
//	  initialzing_time += GetCurrentTimeSec();
	  // End by Johnpzh

    for (int r = 0; r < V; ++r) {
	  // Add by Johnpzh
//	  initialzing_time -= GetCurrentTimeSec();
	  // End by Johnpzh
//      if (usd[r]) {
//		  // Add by Johnpzh
////		  initialzing_time += GetCurrentTimeSec();
//		  // End by Johnpzh
//		  continue;
//	  }
      index_t &idx_r = index_[inv[r]];
      const std::pair<std::vector<int>, std::vector<uint8_t> >
          &tmp_idx_r = tmp_idx[r];
      for (size_t i = 0; i < tmp_idx_r.first.size(); ++i) {
        dst_r[tmp_idx_r.first[i]] = tmp_idx_r.second[i];
      }

      int que_t0 = 0, que_t1 = 0, que_h = 0;
      que[que_h++] = r;
      // vis[r] = true;
      que_t1 = que_h;
	  // Add by Johnpzh
//	  initialzing_time += GetCurrentTimeSec();
//	  adding_ins_count.measure_start();
	  // End by Johnpzh
      using dist_vid_pair = pair<uint8_t, int>;
      priority_queue< dist_vid_pair, vector<dist_vid_pair> , greater<dist_vid_pair> > que;
      que.push_back(make_pair(0, r));
	  dst_r[r] = 0;
      while (que is not empty) {
        int v = vertex in que with min dst_r[v];
        uint8_t v_dist = distance of v in que; // need to check later
        que.pop(); // remove v from que
        if (v_dist > dst_r[v]) {
          continue; // In case some vertices are added twice into que
        }
        std::pair<std::vector<int>, std::vector<uint8_t> >
              &tmp_idx_v = tmp_idx[v];
          index_t &idx_v = index_[inv[v]];

        // Prefetch
        _mm_prefetch(&tmp_idx_v.first[0], _MM_HINT_T0);
        _mm_prefetch(&tmp_idx_v.second[0], _MM_HINT_T0);
        if (usd[v]) {
          continue;
        }
        // Prune
        for (size_t i = 0; i < tmp_idx_v.first.size(); ++i) {
            int w = tmp_idx_v.first[i];
            int td = tmp_idx_v.second[i] + dst_r[w];
            if (td <= v_dist) { // Prune?
              goto pruned;
            }
        }

        // Traverse
        tmp_idx_v.first .back() = r;
        tmp_idx_v.second.back() = d;
        tmp_idx_v.first .push_back(V);
        tmp_idx_v.second.push_back(INF8);
        for (size_t i = 0; i < adj[v].size(); ++i) {
          int w = adj[v][i];
          uint8_t tmp_d = v_dist + dist(v, w);
          if (tmp_d < dst_r[w]) {
            dst_r[w] = tmp_d;
            que.push_back(make_pair(tmp_d, w));
          }
          // if (!vis[w]) {
          //   que[que_h++] = w;
          //   vis[w] = true;
          // }
        }
        pruned:
          {}
      }
      ///////////////////////////////////////////////
      for (int d = 0; que_t0 < que_h; ++d) {
        for (int que_i = que_t0; que_i < que_t1; ++que_i) {
		  // Add by Johnpzh
//		  initialzing_time -= GetCurrentTimeSec();
//			initializing_ins_count.measure_start();
		  // End by Johnpzh
          int v = que[que_i];
          std::pair<std::vector<int>, std::vector<uint8_t> >
              &tmp_idx_v = tmp_idx[v];
          index_t &idx_v = index_[inv[v]];

          // Prefetch
//          _mm_prefetch(&idx_v.bpspt_d[0], _MM_HINT_T0);
//          _mm_prefetch(&idx_v.bpspt_s[0][0], _MM_HINT_T0);
          _mm_prefetch(&tmp_idx_v.first[0], _MM_HINT_T0);
          _mm_prefetch(&tmp_idx_v.second[0], _MM_HINT_T0);

		  // Add by Johnpzh
//		  initialzing_time += GetCurrentTimeSec();
//		  initializing_ins_count.measure_stop();
		  // End by Johnpzh
          // Prune?
		  if (usd[v]) {
			  continue;
		  }
		  // Add by Johnpzh
		  ++total_check_count;
//		  adding_time -= GetCurrentTimeSec();
//		  bp_checking_ins_count.measure_start();
		  // End by Johnpzh
//          for (int i = 0; i < kNumBitParallelRoots; ++i) {
//            int td = idx_r.bpspt_d[i] + idx_v.bpspt_d[i];
//            if (td - 2 <= d) {
//              td +=
//                  (idx_r.bpspt_s[i][0] & idx_v.bpspt_s[i][0]) ? -2 :
//                  ((idx_r.bpspt_s[i][0] & idx_v.bpspt_s[i][1]) |
//                   (idx_r.bpspt_s[i][1] & idx_v.bpspt_s[i][0]))
//                  ? -1 : 0;
//			  if (td <= d) {
//				// Add by Johnpzh
//				++bp_hit_count;
////				bp_checking_ins_count.measure_stop();
//				// End by Johnpzh
//				goto pruned;
//			  }
//            }
//          }
////		  bp_checking_ins_count.measure_stop();
		  ++normal_check_count;
		  distance_query_time -= GetCurrentTimeSec();
		  dist_query_ins_count.measure_start();
          for (size_t i = 0; i < tmp_idx_v.first.size(); ++i) {
            int w = tmp_idx_v.first[i];
            int td = tmp_idx_v.second[i] + dst_r[w];
			if (td <= d) {
				distance_query_time += GetCurrentTimeSec();
				dist_query_ins_count.measure_stop();
				goto pruned;
			}
          }
		  distance_query_time += GetCurrentTimeSec();
		  dist_query_ins_count.measure_stop();

      // Traverse
      tmp_idx_v.first .back() = r;
      tmp_idx_v.second.back() = d;
      tmp_idx_v.first .push_back(V);
      tmp_idx_v.second.push_back(INF8);
      for (size_t i = 0; i < adj[v].size(); ++i) {
        int w = adj[v][i];
        if (!vis[w]) {
          que[que_h++] = w;
          vis[w] = true;
        }
      }
       pruned:
          {}
		  // Add by Johnpzh
//		  adding_time += GetCurrentTimeSec();
		  // End by Johnpzh
        }

        que_t0 = que_t1;
        que_t1 = que_h;
      }
//	  adding_ins_count.measure_stop();

	  // Add by Johnpzh
//	  initialzing_time -= GetCurrentTimeSec();
	  // End by Johnpzh
      for (int i = 0; i < que_h; ++i) vis[que[i]] = false;
      for (size_t i = 0; i < tmp_idx_r.first.size(); ++i) {
        dst_r[tmp_idx_r.first[i]] = INF8;
      }
      usd[r] = true;
	  // Add by Johnpzh
//	  initialzing_time += GetCurrentTimeSec();
	  // End by Johnpzh
    }

	  // Add by Johnpzh
//	  updating_time -= GetCurrentTimeSec();
	  // End by Johnpzh
    for (int v = 0; v < V; ++v) {
      int k = tmp_idx[v].first.size();
      index_[inv[v]].spt_v = (uint32_t*)memalign(64, k * sizeof(uint32_t));
      index_[inv[v]].spt_d = (uint8_t *)memalign(64, k * sizeof(uint8_t ));
      if (!index_[inv[v]].spt_v || !index_[inv[v]].spt_d) {
        Free();
        return false;
      }
      for (int i = 0; i < k; ++i) index_[inv[v]].spt_v[i] = tmp_idx[v].first[i];
      for (int i = 0; i < k; ++i) index_[inv[v]].spt_d[i] = tmp_idx[v].second[i];
      tmp_idx[v].first.clear();
      tmp_idx[v].second.clear();
    }
	  // Add by Johnpzh
//	  updating_time += GetCurrentTimeSec();
	  // End by Johnpzh
  }

  time_indexing_ += GetCurrentTimeSec();
// Add by Johnpzh
//	cache_misses.measure_stop();
// End by Johnpzh
  return true;
}

template<int kNumBitParallelRoots>
int PrunedLandmarkLabeling<kNumBitParallelRoots>
::QueryDistance(int v, int w) {
  if (v >= num_v_ || w >= num_v_) return v == w ? 0 : INT_MAX;

  const index_t &idx_v = index_[v];
  const index_t &idx_w = index_[w];
  int d = INF8;

  _mm_prefetch(&idx_v.spt_v[0], _MM_HINT_T0);
  _mm_prefetch(&idx_w.spt_v[0], _MM_HINT_T0);
  _mm_prefetch(&idx_v.spt_d[0], _MM_HINT_T0);
  _mm_prefetch(&idx_w.spt_d[0], _MM_HINT_T0);

//  for (int i = 0; i < kNumBitParallelRoots; ++i) {
//    int td = idx_v.bpspt_d[i] + idx_w.bpspt_d[i];
//    if (td - 2 <= d) {
//      td +=
//          (idx_v.bpspt_s[i][0] & idx_w.bpspt_s[i][0]) ? -2 :
//          ((idx_v.bpspt_s[i][0] & idx_w.bpspt_s[i][1]) | (idx_v.bpspt_s[i][1] & idx_w.bpspt_s[i][0]))
//          ? -1 : 0;
//
//      if (td < d) d = td;
//    }
//  }
  for (int i1 = 0, i2 = 0; ; ) {
    int v1 = idx_v.spt_v[i1], v2 = idx_w.spt_v[i2];
    if (v1 == v2) {
      if (v1 == num_v_) break;  // Sentinel
      int td = idx_v.spt_d[i1] + idx_w.spt_d[i2];
      if (td < d) d = td;
      ++i1;
      ++i2;
    } else {
      i1 += v1 < v2 ? 1 : 0;
      i2 += v1 > v2 ? 1 : 0;
    }
  }

  if (d >= INF8 - 2) d = INT_MAX;
  return d;
}

template<int kNumBitParallelRoots>
bool PrunedLandmarkLabeling<kNumBitParallelRoots>
::LoadIndex(const char *filename) {
  std::ifstream ifs(filename);
  return ifs && LoadIndex(ifs);
}

template<int kNumBitParallelRoots>
bool PrunedLandmarkLabeling<kNumBitParallelRoots>
::LoadIndex(std::istream &ifs) {
  Free();

  int32_t num_v, num_bpr;
  ifs.read((char*)&num_v,   sizeof(num_v));
  ifs.read((char*)&num_bpr, sizeof(num_bpr));
  num_v_ = num_v;
  if (ifs.bad() || kNumBitParallelRoots != num_bpr) {
    num_v_ = 0;
    return false;
  }

  index_ = (index_t*)memalign(64, num_v * sizeof(index_t));
  if (index_ == NULL) {
    num_v_ = 0;
    return false;
  }
  for (int v = 0; v < num_v_; ++v) {
    index_[v].spt_v = NULL;
    index_[v].spt_d = NULL;
  }

  for (int v = 0; v < num_v_; ++v) {
    index_t &idx = index_[v];

//    for (int i = 0; i < kNumBitParallelRoots; ++i) {
//      ifs.read((char*)&idx.bpspt_d[i]   , sizeof(idx.bpspt_d[i]   ));
//      ifs.read((char*)&idx.bpspt_s[i][0], sizeof(idx.bpspt_s[i][0]));
//      ifs.read((char*)&idx.bpspt_s[i][1], sizeof(idx.bpspt_s[i][1]));
//    }

    int32_t s;
    ifs.read((char*)&s, sizeof(s));
    if (ifs.bad()) {
      Free();
      return false;
    }

    idx.spt_v = (uint32_t*)memalign(64, s * sizeof(uint32_t));
    idx.spt_d = (uint8_t *)memalign(64, s * sizeof(uint8_t ));
    if (!idx.spt_v || !idx.spt_d) {
      Free();
      return false;
    }

    for (int i = 0; i < s; ++i) {
      ifs.read((char*)&idx.spt_v[i], sizeof(idx.spt_v[i]));
      ifs.read((char*)&idx.spt_d[i], sizeof(idx.spt_d[i]));
    }
  }

  return ifs.good();
}

template<int kNumBitParallelRoots>
bool PrunedLandmarkLabeling<kNumBitParallelRoots>
::StoreIndex(const char *filename) {
  std::ofstream ofs(filename);
  return ofs && StoreIndex(ofs);
}

template<int kNumBitParallelRoots>
bool PrunedLandmarkLabeling<kNumBitParallelRoots>
::StoreIndex(std::ostream &ofs) {
  uint32_t num_v = num_v_, num_bpr = kNumBitParallelRoots;
  ofs.write((const char*)&num_v,   sizeof(num_v));
  ofs.write((const char*)&num_bpr, sizeof(num_bpr));

  for (int v = 0; v < num_v_; ++v) {
    index_t &idx = index_[v];

//    for (int i = 0; i < kNumBitParallelRoots; ++i) {
//      int8_t d = idx.bpspt_d[i];
//      uint64_t a = idx.bpspt_s[i][0];
//      uint64_t b = idx.bpspt_s[i][1];
//      ofs.write((const char*)&d, sizeof(d));
//      ofs.write((const char*)&a, sizeof(a));
//      ofs.write((const char*)&b, sizeof(b));
//    }

    int32_t s;
    for (s = 1; idx.spt_v[s - 1] != num_v; ++s) continue;  // Find the sentinel
    ofs.write((const char*)&s, sizeof(s));
    for (int i = 0; i < s; ++i) {
      int32_t l = idx.spt_v[i];
      int8_t  d = idx.spt_d[i];
      ofs.write((const char*)&l, sizeof(l));
      ofs.write((const char*)&d, sizeof(d));
    }
  }

  return ofs.good();
}

template<int kNumBitParallelRoots>
void PrunedLandmarkLabeling<kNumBitParallelRoots>
::Free() {
  for (int v = 0; v < num_v_; ++v) {
    free(index_[v].spt_v);
    free(index_[v].spt_d);
  }
  free(index_);
  index_ = NULL;
  num_v_ = 0;
}

template<int kNumBitParallelRoots>
void PrunedLandmarkLabeling<kNumBitParallelRoots>
::PrintStatistics() {
  std::cout << "bit-parallel_label_size: "   << kNumBitParallelRoots << std::endl;
// Add by Johnpzh
//  printf("BP_Labeling: %f %.2f%%\n", bp_labeling, bp_labeling * 100.0 / time_indexing_);
//  printf("Preparing_time: %f %.2f%%\n", preparing_time, preparing_time * 100.0 / time_indexing_);
//  printf("Initializing: %f %.2f%%\n", initialzing_time, initialzing_time * 100.0 / time_indexing_);
//  printf("Updating: %f %.2f%%\n", updating_time, updating_time * 100.0 / time_indexing_);
//  printf("Adding: %f %.2f%%\n", adding_time, adding_time * 100.0 / time_indexing_);
  	printf("Distance_query_time: %f %.2f%%\n", distance_query_time, distance_query_time * 100.0 / time_indexing_);
	std::cout << "total_check_count: " << total_check_count << std::endl;
	std::cout << "bp_hit_count: " << bp_hit_count << " " << bp_hit_count * 100.0 / total_check_count 
									<< "%" << std::endl;
	printf("normal_check_count: %llu\n", normal_check_count);
//  cache_misses.print();
//	printf("BP_Labeling: "); bp_labeling_ins_count.print();
//	printf("BP_Checking: "); bp_checking_ins_count.print();
//	printf("initializing: ");
//	initializing_ins_count.print();
//	printf("adding: "); adding_ins_count.print();
	printf("distance_query: "); dist_query_ins_count.print();
// End by Johnpzh
//  std::cout << "load time: "     << time_load_     << " seconds" << std::endl;
  std::cout << "indexing_time: " << time_indexing_ << " seconds" << std::endl;

//  double s = 0.0;
//  for (int v = 0; v < num_v_; ++v) {
//    for (int i = 0; index_[v].spt_v[i] != uint32_t(num_v_); ++i) {
//      ++s;
//    }
//  }
//  s /= num_v_;
//  std::cout << "bit-parallel label size: "   << kNumBitParallelRoots << std::endl;
//  std::cout << "average normal label size: " << s << std::endl;
}

#endif  // PRUNED_LANDMARK_LABELING_H_