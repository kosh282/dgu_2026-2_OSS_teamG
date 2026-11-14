// Process-tree WASAPI loopback helper for sherpa_stt.
// Writes mono 16 kHz signed PCM16 to stdout; never creates an audio file.
#ifndef _WIN32_WINNT
#define _WIN32_WINNT 0x0A00
#endif
#include <windows.h>
#include <mmdeviceapi.h>
#include <audioclient.h>
#include <audioclientactivationparams.h>
#include <objidl.h>
#include <propvarutil.h>
#include <io.h>
#include <fcntl.h>
#include <atomic>
#include <cstdint>
#include <iostream>

#pragma comment(lib, "ole32.lib")
#pragma comment(lib, "mmdevapi.lib")

class Activation final : public IActivateAudioInterfaceCompletionHandler {
public:
    Activation() : _refs(1), event(CreateEventW(nullptr, TRUE, FALSE, nullptr)), client(nullptr), result(E_PENDING) {}
    ~Activation() { if (client) client->Release(); if (event) CloseHandle(event); }
    HRESULT STDMETHODCALLTYPE QueryInterface(REFIID iid, void** out) override {
        if (!out) return E_POINTER;
        // ActivateAudioInterfaceAsync rejects non-agile completion handlers with E_ILLEGAL_METHOD_CALL.
        if (iid == __uuidof(IUnknown) || iid == __uuidof(IAgileObject) ||
            iid == __uuidof(IActivateAudioInterfaceCompletionHandler)) {
            *out = static_cast<IActivateAudioInterfaceCompletionHandler*>(this);
            AddRef(); return S_OK;
        }
        *out = nullptr; return E_NOINTERFACE;
    }
    ULONG STDMETHODCALLTYPE AddRef() override { return static_cast<ULONG>(InterlockedIncrement(&_refs)); }
    ULONG STDMETHODCALLTYPE Release() override {
        ULONG refs = static_cast<ULONG>(InterlockedDecrement(&_refs));
        if (!refs) delete this;
        return refs;
    }
    HRESULT STDMETHODCALLTYPE ActivateCompleted(IActivateAudioInterfaceAsyncOperation* operation) override {
        IUnknown* object = nullptr;
        HRESULT activation = E_FAIL;
        result = operation->GetActivateResult(&activation, &object);
        if (SUCCEEDED(result)) result = activation;
        if (SUCCEEDED(result)) result = object->QueryInterface(IID_PPV_ARGS(&client));
        if (object) object->Release();
        SetEvent(event);
        return S_OK;
    }
    bool wait() { return event && WaitForSingleObject(event, 10000) == WAIT_OBJECT_0 && SUCCEEDED(result) && client; }
    HANDLE event;
    IAudioClient* client;
    HRESULT result;
private:
    volatile LONG _refs;
};

static bool check(HRESULT hr, const char* label) {
    if (SUCCEEDED(hr)) return true;
    std::cerr << "ERROR " << label << " 0x" << std::hex << static_cast<unsigned long>(hr) << std::dec << "\n";
    return false;
}

int wmain(int argc, wchar_t** argv) {
    if (argc != 2) { std::cerr << "ERROR expected-process-id\n"; return 2; }
    wchar_t* end = nullptr;
    unsigned long pid = wcstoul(argv[1], &end, 10);
    if (!pid || !end || *end) { std::cerr << "ERROR invalid-process-id\n"; return 2; }
    if (FAILED(CoInitializeEx(nullptr, COINIT_MULTITHREADED))) { std::cerr << "ERROR COM-init\n"; return 3; }
    _setmode(_fileno(stdout), _O_BINARY);

    auto* activation = new Activation();
    AUDIOCLIENT_ACTIVATION_PARAMS params{};
    params.ActivationType = AUDIOCLIENT_ACTIVATION_TYPE_PROCESS_LOOPBACK;
    params.ProcessLoopbackParams.TargetProcessId = pid;
    params.ProcessLoopbackParams.ProcessLoopbackMode = PROCESS_LOOPBACK_MODE_INCLUDE_TARGET_PROCESS_TREE;
    PROPVARIANT value{};
    value.vt = VT_BLOB;
    value.blob.cbSize = sizeof(params);
    value.blob.pBlobData = reinterpret_cast<BYTE*>(&params);
    IActivateAudioInterfaceAsyncOperation* operation = nullptr;
    HRESULT hr = ActivateAudioInterfaceAsync(VIRTUAL_AUDIO_DEVICE_PROCESS_LOOPBACK, __uuidof(IAudioClient),
                                               &value, activation, &operation);
    if (!check(hr, "activate") || !activation->wait()) {
        if (SUCCEEDED(hr)) check(activation->result, "activation-result");
        if (operation) operation->Release();
        activation->Release(); CoUninitialize(); return 4;
    }
    if (operation) operation->Release();

    WAVEFORMATEX format{};
    format.wFormatTag = WAVE_FORMAT_PCM;
    format.nChannels = 1;
    format.nSamplesPerSec = 16000;
    format.wBitsPerSample = 16;
    format.nBlockAlign = 2;
    format.nAvgBytesPerSec = format.nSamplesPerSec * format.nBlockAlign;
    const DWORD flags = AUDCLNT_STREAMFLAGS_LOOPBACK | AUDCLNT_STREAMFLAGS_AUTOCONVERTPCM |
                        AUDCLNT_STREAMFLAGS_SRC_DEFAULT_QUALITY;
    if (!check(activation->client->Initialize(AUDCLNT_SHAREMODE_SHARED, flags, 0, 0, &format, nullptr), "initialize")) {
        activation->Release(); CoUninitialize(); return 5;
    }
    IAudioCaptureClient* capture = nullptr;
    if (!check(activation->client->GetService(IID_PPV_ARGS(&capture)), "capture-service") ||
        !check(activation->client->Start(), "start")) {
        if (capture) capture->Release(); activation->Release(); CoUninitialize(); return 6;
    }
    std::cerr << "READY\n";

    bool failed = false;
    while (true) {
        UINT32 packetFrames = 0;
        hr = capture->GetNextPacketSize(&packetFrames);
        if (FAILED(hr)) { check(hr, "next-packet"); failed = true; break; }
        if (!packetFrames) { Sleep(3); continue; }
        BYTE* data = nullptr;
        UINT32 frames = 0;
        DWORD bufferFlags = 0;
        hr = capture->GetBuffer(&data, &frames, &bufferFlags, nullptr, nullptr);
        if (FAILED(hr)) { check(hr, "get-buffer"); failed = true; break; }
        if (bufferFlags & AUDCLNT_BUFFERFLAGS_DATA_DISCONTINUITY) std::cerr << "DISCONTINUITY\n";
        if (bufferFlags & AUDCLNT_BUFFERFLAGS_SILENT) {
            static const int16_t silence[2048] = {};
            UINT32 remaining = frames;
            while (remaining) {
                UINT32 n = remaining < 2048 ? remaining : 2048;
                std::cout.write(reinterpret_cast<const char*>(silence), n * sizeof(int16_t));
                remaining -= n;
            }
        } else {
            std::cout.write(reinterpret_cast<const char*>(data), frames * sizeof(int16_t));
        }
        std::cout.flush();
        if (!check(capture->ReleaseBuffer(frames), "release-buffer")) { failed = true; break; }
        if (!std::cout) { failed = true; break; }
    }
    activation->client->Stop();
    capture->Release(); activation->Release(); CoUninitialize();
    return failed ? 7 : 0;
}
