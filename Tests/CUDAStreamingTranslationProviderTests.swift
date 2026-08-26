import Foundation
import Testing
@testable import UniversalInterpreter

@Suite("CUDA streaming provider configuration")
struct CUDAStreamingTranslationProviderTests {
    @Test func explicitDisableWins() {
        #expect(CUDAStreamingServerConfiguration.configured(environment: [
            "AI_INTERPRETER_DISABLE_GPU_STREAMING": "1",
            "AI_INTERPRETER_STREAMING_SERVER_URL": "https://gpu.example.test",
        ]) == nil)
    }

    @Test func rejectsNonHTTPServer() {
        #expect(CUDAStreamingServerConfiguration.configured(environment: [
            "AI_INTERPRETER_STREAMING_SERVER_URL": "file:///tmp/server",
        ]) == nil)
    }

    @Test func buildsAuthenticatedBilingualWebSocketURL() throws {
        let configuration = try #require(
            CUDAStreamingServerConfiguration.configured(environment: [
                "AI_INTERPRETER_STREAMING_SERVER_URL": "https://gpu.example.test/base",
                "AI_INTERPRETER_STREAMING_SERVER_TOKEN": "secret",
            ])
        )
        #expect(configuration.authorization == "secret")
        let url = try configuration.webSocketURL(sourceLanguage: .korean)
        #expect(url.scheme == "wss")
        #expect(url.path == "/base/v1/asr/stream")
        #expect(url.query?.contains("language=ko") == true)
        #expect(url.query?.contains("sample_rate=16000") == true)
        #expect(url.query?.contains("token=secret") == true)
    }

    @Test func configuresIndependentVoiceService() throws {
        let configuration = try #require(
            CUDAStreamingServerConfiguration.configured(environment: [
                "AI_INTERPRETER_STREAMING_SERVER_URL": "https://asr.example.test",
                "AI_INTERPRETER_STREAMING_SERVER_TOKEN": "asr-secret",
                "AI_INTERPRETER_TTS_SERVER_URL": "https://voice.example.test",
                "AI_INTERPRETER_TTS_SERVER_TOKEN": "voice-secret",
            ])
        )
        #expect(configuration.voiceEndpoint.absoluteString == "https://voice.example.test")
        #expect(configuration.voiceAuthorization == "voice-secret")
        #expect(configuration.requiresVoicePortHandoff == false)
    }
}
