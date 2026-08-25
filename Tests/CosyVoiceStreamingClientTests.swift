import Foundation
import Testing
@testable import UniversalInterpreter

@Test func cosyVoiceConfigurationRequiresHTTPURL() {
    #expect(CosyVoiceStreamingClient.configured(environment: [:]) == nil)
    #expect(CosyVoiceStreamingClient.configured(environment: [
        "AI_INTERPRETER_TTS_SERVER_URL": "not a url"
    ]) == nil)
    #expect(CosyVoiceStreamingClient.configured(environment: [
        "AI_INTERPRETER_TTS_SERVER_URL": "https://voice.example.test"
    ]) != nil)
}

@Test func cosyVoiceDecodesLittleEndianPCM16() {
    var values = [Int16.min, 0, Int16.max].map { $0.littleEndian }
    let data = values.withUnsafeBytes { Data($0) }
    let decoded = CosyVoiceStreamingClient.decodePCM16(data)
    #expect(decoded.count == 3)
    #expect(decoded[0] < -0.99)
    #expect(decoded[1] == 0)
    #expect(decoded[2] > 0.99)
}

@Test func cosyVoiceCanShareCUDAStreamingConfiguration() throws {
    let configuration = try #require(
        CUDAStreamingServerConfiguration.configured(environment: [
            "AI_INTERPRETER_STREAMING_SERVER_URL": "https://gpu.example.test/base",
            "AI_INTERPRETER_STREAMING_SERVER_TOKEN": "secret",
        ])
    )
    let client = CosyVoiceStreamingClient.configured(from: configuration)
    #expect(client.endpoint.absoluteString == "https://gpu.example.test/base")
}
