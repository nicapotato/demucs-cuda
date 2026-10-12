// Windows Media Foundation fallback decoder (m4a/aac/wma/...). Kept out of audio.c so the
// Windows SDK headers are not mixed with the single-header decoders' macros.
#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <mfapi.h>
#include <mfidl.h>
#include <mfreadwrite.h>

#include <stdlib.h>
#include <string.h>

#include "audio_mf.h"

extern "C" int dc_mf_decode(const char *path_utf8, float **pcm_out, int *channels,
                            long long *frames, int *sample_rate) {
  *pcm_out = NULL;
  int wlen = MultiByteToWideChar(CP_UTF8, 0, path_utf8, -1, NULL, 0);
  if (wlen <= 0) {
    return 0;
  }
  wchar_t *wpath = (wchar_t *)malloc((size_t)wlen * sizeof(wchar_t));
  if (!wpath || !MultiByteToWideChar(CP_UTF8, 0, path_utf8, -1, wpath, wlen)) {
    free(wpath);
    return 0;
  }
  HRESULT hr = CoInitializeEx(NULL, COINIT_MULTITHREADED);
  bool com_init = SUCCEEDED(hr);
  if (FAILED(MFStartup(MF_VERSION, MFSTARTUP_LITE))) {
    if (com_init) {
      CoUninitialize();
    }
    free(wpath);
    return 0;
  }
  int ok = 0;
  IMFSourceReader *reader = NULL;
  IMFMediaType *type = NULL;
  IMFMediaType *cur = NULL;
  float *pcm = NULL;
  size_t len = 0, cap = 0;
  UINT32 ch = 0, sr = 0;
  const DWORD stream = (DWORD)MF_SOURCE_READER_FIRST_AUDIO_STREAM;

  if (FAILED(MFCreateSourceReaderFromURL(wpath, NULL, &reader))) {
    goto done;
  }
  reader->SetStreamSelection((DWORD)MF_SOURCE_READER_ALL_STREAMS, FALSE);
  reader->SetStreamSelection(stream, TRUE);
  if (FAILED(MFCreateMediaType(&type))) {
    goto done;
  }
  type->SetGUID(MF_MT_MAJOR_TYPE, MFMediaType_Audio);
  type->SetGUID(MF_MT_SUBTYPE, MFAudioFormat_Float);
  if (FAILED(reader->SetCurrentMediaType(stream, NULL, type))) {
    goto done;
  }
  if (FAILED(reader->GetCurrentMediaType(stream, &cur))) {
    goto done;
  }
  cur->GetUINT32(MF_MT_AUDIO_NUM_CHANNELS, &ch);
  cur->GetUINT32(MF_MT_AUDIO_SAMPLES_PER_SECOND, &sr);
  if (ch == 0 || sr == 0) {
    goto done;
  }
  for (;;) {
    DWORD flags = 0;
    IMFSample *sample = NULL;
    if (FAILED(reader->ReadSample(stream, 0, NULL, &flags, NULL, &sample))) {
      break;
    }
    if (sample) {
      IMFMediaBuffer *mb = NULL;
      if (SUCCEEDED(sample->ConvertToContiguousBuffer(&mb))) {
        BYTE *p = NULL;
        DWORD nbytes = 0;
        if (SUCCEEDED(mb->Lock(&p, NULL, &nbytes))) {
          size_t nf = nbytes / sizeof(float);
          if (len + nf > cap) {
            cap = (len + nf) * 2;
            float *np = (float *)realloc(pcm, cap * sizeof(float));
            if (!np) {
              mb->Unlock();
              mb->Release();
              sample->Release();
              goto done;
            }
            pcm = np;
          }
          memcpy(pcm + len, p, nf * sizeof(float));
          len += nf;
          mb->Unlock();
        }
        mb->Release();
      }
      sample->Release();
    }
    if (flags & (MF_SOURCE_READERF_ENDOFSTREAM | MF_SOURCE_READERF_ERROR)) {
      break;
    }
  }
  if (len >= ch) {
    *pcm_out = pcm;
    pcm = NULL;
    *channels = (int)ch;
    *frames = (long long)(len / ch);
    *sample_rate = (int)sr;
    ok = 1;
  }
done:
  free(pcm);
  free(wpath);
  if (cur) {
    cur->Release();
  }
  if (type) {
    type->Release();
  }
  if (reader) {
    reader->Release();
  }
  MFShutdown();
  if (com_init) {
    CoUninitialize();
  }
  return ok;
}
#endif
