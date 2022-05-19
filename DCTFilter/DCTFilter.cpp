/*
    MIT License

    Copyright (c) 2017 HolyWu

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.
*/

#include <algorithm>
#include <cmath>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <string>
#include <thread>
#include <type_traits>
#include <unordered_map>
#include <vector>

#include <VapourSynth.h>
#include <VSHelper.h>

#include <fftw3.h>

struct DCTFilterData {
    VSNodeRef * node;
    const VSVideoInfo * vi;
    bool process[3];
    int peak;
    int n;
    std::vector<float> qps;
    std::vector<float> factors;
    fftwf_plan dct, idct;
    std::unordered_map<std::thread::id, float *> buffer;
    std::shared_mutex buffer_lock;
};

template<typename T>
static void process(const VSFrameRef * src, VSFrameRef * dst, DCTFilterData * d, const VSAPI * vsapi) noexcept {
    const auto threadId = std::this_thread::get_id();
    float * VS_RESTRICT buffer;
    {
        d->buffer_lock.lock_shared();
        buffer = d->buffer[threadId];
        d->buffer_lock.unlock_shared();
    }

    for (int plane = 0; plane < d->vi->format->numPlanes; plane++) {
        if (d->process[plane]) {
            const int width = vsapi->getFrameWidth(src, plane);
            const int height = vsapi->getFrameHeight(src, plane);
            const int stride = vsapi->getStride(src, plane) / sizeof(T);
            const T * srcp = reinterpret_cast<const T *>(vsapi->getReadPtr(src, plane));
            T * VS_RESTRICT dstp = reinterpret_cast<T *>(vsapi->getWritePtr(dst, plane));

            const int n = d->n;
            for (int y = 0; y < height; y += n) {
                for (int x = 0; x < width; x += n) {
                    for (int yy = 0; yy < n; yy++) {
                        const T * input = srcp + stride * yy + x;
                        float * VS_RESTRICT output = buffer + n * yy;

                        for (int xx = 0; xx < n; xx++)
                            output[xx] = input[xx];
                    }

                    fftwf_execute_r2r(d->dct, buffer, buffer);

                    for (int i = 0; i < n * n; i++) {
                        buffer[i] *= d->factors[i];
                        if (d->qps[i] > 0.0f) {
                            buffer[i] -= fmodf(buffer[i], d->qps[i]);
                        }
                    }

                    fftwf_execute_r2r(d->idct, buffer, buffer);

                    for (int yy = 0; yy < n; yy++) {
                        const float * input = buffer + n * yy;
                        T * VS_RESTRICT output = dstp + stride * yy + x;

                        for (int xx = 0; xx < n; xx++) {
                            if (std::is_integral<T>::value)
                                output[xx] = std::min(std::max(static_cast<int>(input[xx] + 0.5f), 0), d->peak);
                            else
                                output[xx] = input[xx];
                        }
                    }
                }

                srcp += stride * n;
                dstp += stride * n;
            }
        }
    }
}

static void VS_CC dctfilterInit(VSMap *in, VSMap *out, void **instanceData, VSNode *node, VSCore *core, const VSAPI *vsapi) {
    DCTFilterData * d = static_cast<DCTFilterData *>(*instanceData);
    vsapi->setVideoInfo(d->vi, 1, node);
}

static const VSFrameRef *VS_CC dctfilterGetFrame(int n, int activationReason, void **instanceData, void **frameData, VSFrameContext *frameCtx, VSCore *core, const VSAPI *vsapi) {
    DCTFilterData * d = static_cast<DCTFilterData *>(*instanceData);

    if (activationReason == arInitial) {
        vsapi->requestFrameFilter(n, d->node, frameCtx);
    } else if (activationReason == arAllFramesReady) {
        try {
            auto threadId = std::this_thread::get_id();

            if (!d->buffer.count(threadId)) {
                float * buffer = fftwf_alloc_real(d->n * d->n);
                if (!buffer)
                    throw std::string{ "malloc failure (buffer)" };

                {
                    std::lock_guard<std::shared_mutex> l(d->buffer_lock);
                    d->buffer.emplace(threadId, buffer);
                }
            }
        } catch (const std::string & error) {
            vsapi->setFilterError(("DCTFilter: " + error).c_str(), frameCtx);
            return nullptr;
        }

        const VSFrameRef * src = vsapi->getFrameFilter(n, d->node, frameCtx);
        const VSFrameRef * fr[] = { d->process[0] ? nullptr : src, d->process[1] ? nullptr : src, d->process[2] ? nullptr : src };
        const int pl[] = { 0, 1, 2 };
        VSFrameRef * dst = vsapi->newVideoFrame2(d->vi->format, d->vi->width, d->vi->height, fr, pl, src, core);

        if (d->vi->format->bytesPerSample == 1)
            process<uint8_t>(src, dst, d, vsapi);
        else if (d->vi->format->bytesPerSample == 2)
            process<uint16_t>(src, dst, d, vsapi);
        else
            process<float>(src, dst, d, vsapi);

        vsapi->freeFrame(src);
        return dst;
    }

    return nullptr;
}

static void VS_CC dctfilterFree(void *instanceData, VSCore *core, const VSAPI *vsapi) {
    DCTFilterData * d = static_cast<DCTFilterData *>(instanceData);

    vsapi->freeNode(d->node);

    fftwf_destroy_plan(d->dct);
    fftwf_destroy_plan(d->idct);

    for (auto & iter : d->buffer)
        fftwf_free(iter.second);

    delete d;
}

static void VS_CC dctfilterCreate(const VSMap *in, VSMap *out, void *userData, VSCore *core, const VSAPI *vsapi) {
    std::unique_ptr<DCTFilterData> d{ new DCTFilterData{} };

    d->node = vsapi->propGetNode(in, "clip", 0, nullptr);
    d->vi = vsapi->getVideoInfo(d->node);

    int padWidth = 0, padHeight = 0;

    try {
        int err;
        int n = vsapi->propGetInt(in, "n", 0, &err);
        if (err != 0) n = 8;
        if (n < 0 || (n & (n - 1)) != 0)
            throw std::string{ "n must be power of two and > 1" };
        d->n = n;

        padWidth = (d->vi->width & (2*n-1)) ? 2*n - d->vi->width % (2*n) : 0;
        padHeight = (d->vi->height & (2*n-1)) ? 2*n - d->vi->height % (2*n) : 0;

        if (!isConstantFormat(d->vi) || (d->vi->format->sampleType == stInteger && d->vi->format->bitsPerSample > 16) ||
            (d->vi->format->sampleType == stFloat && d->vi->format->bitsPerSample != 32))
            throw std::string{ "only constant format 8-16 bit integer and 32 bit float input supported" };

        const double * factors = vsapi->propGetFloatArray(in, "factors", nullptr);

        const int m = vsapi->propNumElements(in, "planes");

        for (int i = 0; i < 3; i++)
            d->process[i] = (m <= 0);

        for (int i = 0; i < m; i++) {
            const int n = int64ToIntS(vsapi->propGetInt(in, "planes", i, nullptr));

            if (n < 0 || n >= d->vi->format->numPlanes)
                throw std::string{ "plane index out of range" };

            if (d->process[n])
                throw std::string{ "plane specified twice" };

            d->process[n] = true;
        }

        const int nfactors = vsapi->propNumElements(in, "factors");
        if (nfactors != d->n && nfactors != d->n * d->n)
            throw std::string{ "the number of factors must be equal to either n or n*n" };

        for (int i = 0; i < nfactors; i++) {
            if (factors[i] < 0. || factors[i] > 1.)
                throw std::string{ "factor must be between 0.0 and 1.0 (inclusive)" };
        }

        VSCoreInfo coreinfo;
        vsapi->getCoreInfo2(core, &coreinfo);
        const unsigned numThreads = coreinfo.numThreads;
        d->buffer.reserve(numThreads);

        if (d->vi->format->sampleType == stInteger)
            d->peak = (1 << d->vi->format->bitsPerSample) - 1;

        if (padWidth || padHeight) {
            VSMap * args = vsapi->createMap();
            vsapi->propSetNode(args, "clip", d->node, paReplace);
            vsapi->freeNode(d->node);
            vsapi->propSetInt(args, "width", d->vi->width + padWidth, paReplace);
            vsapi->propSetInt(args, "height", d->vi->height + padHeight, paReplace);
            vsapi->propSetFloat(args, "src_width", d->vi->width + padWidth, paReplace);
            vsapi->propSetFloat(args, "src_height", d->vi->height + padHeight, paReplace);

            VSMap * ret = vsapi->invoke(vsapi->getPluginById("com.vapoursynth.resize", core), "Point", args);
            if (vsapi->getError(ret)) {
                vsapi->setError(out, vsapi->getError(ret));
                vsapi->freeMap(args);
                vsapi->freeMap(ret);
                return;
            }

            d->node = vsapi->propGetNode(ret, "clip", 0, nullptr);
            d->vi = vsapi->getVideoInfo(d->node);
            vsapi->freeMap(args);
            vsapi->freeMap(ret);
        }

        d->factors.resize(d->n * d->n);
        const auto norm = 1.f / (d->n * d->n * 4);
        if (nfactors != d->n) {
            for (int i = 0; i < nfactors; i++)
                d->factors[i] = static_cast<float>(factors[i]) * norm;
        } else {
            for (int y = 0; y < d->n; y++) {
                for (int x = 0; x < d->n; x++)
                    d->factors[d->n * y + x] = static_cast<float>(factors[y] * factors[x]) * norm;
            }
        }

        d->qps.resize(d->n * d->n);
        const int nqps = vsapi->propNumElements(in, "qps");
        if (nqps > 0) {
            const double * qps = vsapi->propGetFloatArray(in, "qps", nullptr);
            if (nqps != d->n && nqps != d->n * d->n)
                throw std::string{ "the number of qps must be equal to either n or n*n" };

            if (nqps != d->n) {
                for (int i = 0; i < nqps; i++)
                    d->qps[i] = static_cast<float>(qps[i]);
            } else {
                for (int y = 0; y < d->n; y++) {
                    for (int x = 0; x < d->n; x++)
                        d->qps[d->n * y + x] = static_cast<float>(qps[y] * qps[x]);
                }
            }
            if (d->vi->format->sampleType == stInteger) {
                for (int i = 0; i < d->n * d->n; i++) {
                    d->qps[i] *= (1 << d->vi->format->bitsPerSample) - 1;
                }
            }
            d->qps[0] *= 2;
            for (int i = 1; i < d->n; i++) {
                d->qps[i] *= std::sqrt(2.0f);
            }
            for (int i = 1; i < d->n; i++) {
                d->qps[d->n * i] *= std::sqrt(2.0f);
            }
        }

        float * buffer = fftwf_alloc_real(d->n * d->n);
        if (!buffer)
            throw std::string{ "malloc failure (buffer)" };

        d->dct = fftwf_plan_r2r_2d(d->n, d->n, buffer, buffer, FFTW_REDFT10, FFTW_REDFT10, FFTW_PATIENT);
        d->idct = fftwf_plan_r2r_2d(d->n, d->n, buffer, buffer, FFTW_REDFT01, FFTW_REDFT01, FFTW_PATIENT);

        fftwf_free(buffer);
    } catch (const std::string & error) {
        vsapi->setError(out, ("DCTFilter: " + error).c_str());
        vsapi->freeNode(d->node);
        return;
    }

    vsapi->createFilter(in, out, "DCTFilter", dctfilterInit, dctfilterGetFrame, dctfilterFree, fmParallel, 0, d.release(), core);

    if (padWidth || padHeight) {
        VSNodeRef * node = vsapi->propGetNode(out, "clip", 0, nullptr);
        vsapi->clearMap(out);

        VSMap * args = vsapi->createMap();
        vsapi->propSetNode(args, "clip", node, paReplace);
        vsapi->freeNode(node);
        vsapi->propSetInt(args, "right", padWidth, paReplace);
        vsapi->propSetInt(args, "bottom", padHeight, paReplace);

        VSMap * ret = vsapi->invoke(vsapi->getPluginById("com.vapoursynth.std", core), "Crop", args);
        if (vsapi->getError(ret)) {
            vsapi->setError(out, vsapi->getError(ret));
            vsapi->freeMap(args);
            vsapi->freeMap(ret);
            return;
        }

        node = vsapi->propGetNode(ret, "clip", 0, nullptr);
        vsapi->freeMap(args);
        vsapi->freeMap(ret);
        vsapi->propSetNode(out, "clip", node, paReplace);
        vsapi->freeNode(node);
    }
}

//////////////////////////////////////////
// Init

VS_EXTERNAL_API(void) VapourSynthPluginInit(VSConfigPlugin configFunc, VSRegisterFunction registerFunc, VSPlugin *plugin) {
    configFunc("com.holywu.dctfilter", "dctf", "DCT/IDCT Frequency Suppressor", VAPOURSYNTH_API_VERSION, 1, plugin);
    registerFunc("DCTFilter",
                 "clip:clip;"
                 "factors:float[];"
                 "planes:int[]:opt;"
                 "n:int:opt;"
                 "qps:float[]:opt;",
                 dctfilterCreate, nullptr, plugin);
}
