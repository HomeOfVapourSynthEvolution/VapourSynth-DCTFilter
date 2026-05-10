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
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <string>
#include <thread>
#include <type_traits>
#include <unordered_map>

#include <VapourSynth4.h>
#include <VSHelper4.h>

#include <fftw3.h>

using namespace std::string_literals;

static std::mutex planMutex;

struct DCTFilterData final {
    VSNode* node;
    const VSVideoInfo* vi;
    float factors[64];
    bool process[3];
    fftwf_plan dct, idct;
    int peak;
    void (*filter)(const VSFrame* src, VSFrame* dst, float* VS_RESTRICT buffer, const DCTFilterData* VS_RESTRICT d, const VSAPI* vsapi) noexcept;
};

template<typename pixel_t>
static void filter(const VSFrame* src, VSFrame* dst, float* VS_RESTRICT buffer, const DCTFilterData* VS_RESTRICT d, const VSAPI* vsapi) noexcept {
    for (int plane = 0; plane < d->vi->format.numPlanes; plane++) {
        if (d->process[plane]) {
            const int width = vsapi->getFrameWidth(src, plane);
            const int height = vsapi->getFrameHeight(src, plane);
            const ptrdiff_t stride = vsapi->getStride(src, plane) / sizeof(pixel_t);
            const pixel_t* srcp = reinterpret_cast<const pixel_t*>(vsapi->getReadPtr(src, plane));
            pixel_t* VS_RESTRICT dstp = reinterpret_cast<pixel_t*>(vsapi->getWritePtr(dst, plane));

            for (int y = 0; y < height; y += 8) {
                for (int x = 0; x < width; x += 8) {
                    for (int yy = 0; yy < 8; yy++) {
                        const pixel_t* input = srcp + stride * yy + x;
                        float* VS_RESTRICT output = buffer + 8 * yy;

                        for (int xx = 0; xx < 8; xx++)
                            output[xx] = input[xx] * (1.0f / 256.0f);
                    }

                    fftwf_execute_r2r(d->dct, buffer, buffer);

                    for (int i = 0; i < 64; i++)
                        buffer[i] *= d->factors[i];

                    fftwf_execute_r2r(d->idct, buffer, buffer);

                    for (int yy = 0; yy < 8; yy++) {
                        const float* input = buffer + 8 * yy;
                        pixel_t* VS_RESTRICT output = dstp + stride * yy + x;

                        for (int xx = 0; xx < 8; xx++) {
                            if constexpr (std::is_integral_v<pixel_t>)
                                output[xx] = std::min(std::max(static_cast<int>(input[xx] + 0.5f), 0), d->peak);
                            else
                                output[xx] = input[xx];
                        }
                    }
                }

                srcp += stride * 8;
                dstp += stride * 8;
            }
        }
    }
}

static const VSFrame* VS_CC dctFilterGetFrame(int n, int activationReason, void* instanceData, [[maybe_unused]] void** frameData, VSFrameContext* frameCtx,
                                              VSCore* core, const VSAPI* vsapi) {
    auto d = static_cast<DCTFilterData*>(instanceData);

    if (activationReason == arInitial) {
        vsapi->requestFrameFilter(n, d->node, frameCtx);
    } else if (activationReason == arAllFramesReady) {
        float *buffer = vsh::vsh_aligned_malloc<float>(64 * sizeof(float), 64);

        const VSFrame* src = vsapi->getFrameFilter(n, d->node, frameCtx);
        const VSFrame* fr[] = { d->process[0] ? nullptr : src, d->process[1] ? nullptr : src, d->process[2] ? nullptr : src };
        const int pl[] = { 0, 1, 2 };
        VSFrame* dst = vsapi->newVideoFrame2(&d->vi->format, d->vi->width, d->vi->height, fr, pl, src, core);

        d->filter(src, dst, buffer, d, vsapi);

        vsh::vsh_aligned_free(buffer);
        vsapi->freeFrame(src);
        return dst;
    }

    return nullptr;
}

static void VS_CC dctFilterFree(void* instanceData, [[maybe_unused]] VSCore* core, const VSAPI* vsapi) {
    auto d = static_cast<DCTFilterData*>(instanceData);

    vsapi->freeNode(d->node);

    {
        std::lock_guard<std::mutex> lock(planMutex);
        fftwf_destroy_plan(d->dct);
        fftwf_destroy_plan(d->idct);
    }

    delete d;
}

static void VS_CC dctFilterCreate(const VSMap* in, VSMap* out, [[maybe_unused]] void* userData, VSCore* core, const VSAPI* vsapi) {
    auto d = std::make_unique<DCTFilterData>();

    d->node = vsapi->mapGetNode(in, "clip", 0, nullptr);
    d->vi = vsapi->getVideoInfo(d->node);

    const int padWidth = (d->vi->width & 15) ? 16 - d->vi->width % 16 : 0;
    const int padHeight = (d->vi->height & 15) ? 16 - d->vi->height % 16 : 0;

    try {
        if (!vsh::isConstantVideoFormat(d->vi) ||
            (d->vi->format.sampleType == stInteger && d->vi->format.bitsPerSample > 16) ||
            (d->vi->format.sampleType == stFloat && d->vi->format.bitsPerSample != 32))
            throw "only constant format 8-16 bit integer and 32 bit float input supported"s;

        const double* factors = vsapi->mapGetFloatArray(in, "factors", nullptr);

        const int m = vsapi->mapNumElements(in, "planes");

        for (int i = 0; i < 3; i++)
            d->process[i] = (m <= 0);

        for (int i = 0; i < m; i++) {
            const int n = vsapi->mapGetIntSaturated(in, "planes", i, nullptr);

            if (n < 0 || n >= d->vi->format.numPlanes)
                throw "plane index out of range"s;

            if (d->process[n])
                throw "plane specified twice"s;

            d->process[n] = true;
        }

        if (vsapi->mapNumElements(in, "factors") != 8)
            throw "number of elements in factors must be 8"s;

        for (int i = 0; i < 8; i++) {
            if (factors[i] < 0.0 || factors[i] > 1.0)
                throw "factor must be between 0.0 and 1.0 (inclusive)"s;
        }

        for (int y = 0; y < 8; y++) {
            for (int x = 0; x < 8; x++)
                d->factors[8 * y + x] = static_cast<float>(factors[y] * factors[x]);
        }

        float* buffer = vsh::vsh_aligned_malloc<float>(64 * sizeof(float), 64);

        {
            std::lock_guard<std::mutex> lock(planMutex);
            d->dct = fftwf_plan_r2r_2d(8, 8, buffer, buffer, FFTW_REDFT10, FFTW_REDFT10, FFTW_PATIENT);
            d->idct = fftwf_plan_r2r_2d(8, 8, buffer, buffer, FFTW_REDFT01, FFTW_REDFT01, FFTW_PATIENT);
        }

        vsh::vsh_aligned_free(buffer);

        if (d->vi->format.sampleType == stInteger)
            d->peak = (1 << d->vi->format.bitsPerSample) - 1;

        if (d->vi->format.bytesPerSample == 1)
            d->filter = filter<uint8_t>;
        else if (d->vi->format.bytesPerSample == 2)
            d->filter = filter<uint16_t>;
        else
            d->filter = filter<float>;

        if (padWidth || padHeight) {
            VSMap* args = vsapi->createMap();
            vsapi->mapConsumeNode(args, "clip", d->node, maReplace);
            vsapi->mapSetInt(args, "width", d->vi->width + padWidth, maReplace);
            vsapi->mapSetInt(args, "height", d->vi->height + padHeight, maReplace);
            vsapi->mapSetFloat(args, "src_width", d->vi->width + padWidth, maReplace);
            vsapi->mapSetFloat(args, "src_height", d->vi->height + padHeight, maReplace);

            VSMap* ret = vsapi->invoke(vsapi->getPluginByID(VSH_RESIZE_PLUGIN_ID, core), "Point", args);
            if (vsapi->mapGetError(ret)) {
                vsapi->mapSetError(out, vsapi->mapGetError(ret));
                vsapi->freeMap(args);
                vsapi->freeMap(ret);
                return;
            }

            d->node = vsapi->mapGetNode(ret, "clip", 0, nullptr);
            d->vi = vsapi->getVideoInfo(d->node);
            vsapi->freeMap(args);
            vsapi->freeMap(ret);
        }
    } catch (const std::string& error) {
        vsapi->mapSetError(out, ("DCTFilter: " + error).c_str());
        vsapi->freeNode(d->node);
        return;
    }

    VSFilterDependency deps[] = { {d->node, rpStrictSpatial} };
    vsapi->createVideoFilter(out, "DCTFilter", d->vi, dctFilterGetFrame, dctFilterFree, fmParallel, deps, 1, d.get(), core);
    d.release();

    if (padWidth || padHeight) {
        VSNode* node = vsapi->mapGetNode(out, "clip", 0, nullptr);
        vsapi->clearMap(out);

        VSMap* args = vsapi->createMap();
        vsapi->mapConsumeNode(args, "clip", node, maReplace);
        vsapi->mapSetInt(args, "right", padWidth, maReplace);
        vsapi->mapSetInt(args, "bottom", padHeight, maReplace);

        VSMap* ret = vsapi->invoke(vsapi->getPluginByID(VSH_STD_PLUGIN_ID, core), "Crop", args);
        if (vsapi->mapGetError(ret)) {
            vsapi->mapSetError(out, vsapi->mapGetError(ret));
            vsapi->freeMap(args);
            vsapi->freeMap(ret);
            return;
        }

        node = vsapi->mapGetNode(ret, "clip", 0, nullptr);
        vsapi->freeMap(args);
        vsapi->freeMap(ret);
        vsapi->mapConsumeNode(out, "clip", node, maReplace);
    }
}

//////////////////////////////////////////
// Init

VS_EXTERNAL_API(void) VapourSynthPluginInit2(VSPlugin* plugin, const VSPLUGINAPI* vspapi) {
    vspapi->configPlugin("com.holywu.dctfilter", "dctf", "DCT/IDCT Frequency Suppressor", VS_MAKE_VERSION(3, 2), VAPOURSYNTH_API_VERSION, 0, plugin);

    vspapi->registerFunction("DCTFilter", "clip:vnode;factors:float[];planes:int[]:opt;", "clip:vnode;", dctFilterCreate, nullptr, plugin);
}
