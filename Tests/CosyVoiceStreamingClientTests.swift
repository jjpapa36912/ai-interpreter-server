import Foundation
import Testing
@testable import UniversalInterpreter

private final class TTSJobURLProtocolState: @unchecked Sendable {
    private let lock = NSLock()
    private var storage: [String] = []

    func reset() { lock.withLock { storage = [] } }
    func append(_ path: String) { lock.withLock { storage.append(path) } }
    var paths: [String] { lock.withLock { storage } }
}

private final class TTSJobURLProtocol: URLProtocol, @unchecked Sendable {
    static let state = TTSJobURLProtocolState()

    override class func canInit(with request: URLRequest) -> Bool {
        request.url?.host == "pcmjob.example.test"
    }

    override class func canonicalRequest(for request: URLRequest) -> URLRequest { request }

    override func startLoading() {
        guard let url = request.url else { return }
        Self.state.append(url.path)
        let response: HTTPURLResponse
        let body: Data
        if url.path == "/v1/tts/jobs" {
            body = Data(
                #"{"job_id":"job-1","sample_rate":24000,"transport":"pcm-pull-v1"}"#.utf8
            )
            response = HTTPURLResponse(
                url: url, statusCode: 200, httpVersion: "HTTP/1.1",
                headerFields: ["Content-Type": "application/json"]
            )!
        } else if url.path == "/v1/tts/jobs/job-1/chunks/0" {
            let values = [Int16(-16_000).littleEndian, Int16(16_000).littleEndian]
            body = values.withUnsafeBytes { Data($0) }
            response = HTTPURLResponse(
                url: url, statusCode: 200, httpVersion: "HTTP/1.1",
                headerFields: [
                    "Content-Type": "audio/L16;rate=24000;channels=1",
                    "X-TTS-Done": "1",
                ]
            )!
        } else {
            body = Data()
            response = HTTPURLResponse(
                url: url, statusCode: 404, httpVersion: "HTTP/1.1",
                headerFields: nil
            )!
        }
        client?.urlProtocol(self, didReceive: response, cacheStoragePolicy: .notAllowed)
        client?.urlProtocol(self, didLoad: body)
        client?.urlProtocolDidFinishLoading(self)
    }

    override func stopLoading() {}
}

private final class AudioChunkBox: @unchecked Sendable {
    private let lock = NSLock()
    private var storage: [AudioChunk] = []

    func append(_ chunk: AudioChunk) { lock.withLock { storage.append(chunk) } }
    var chunks: [AudioChunk] { lock.withLock { storage } }
}

@Test func cosyVoiceConfigurationRequiresHTTPURL() {
    #expect(CosyVoiceStreamingClient.configured(environment: [:]) == nil)
    #expect(CosyVoiceStreamingClient.configured(environment: [
        "AI_INTERPRETER_TTS_SERVER_URL": "not a url"
    ]) == nil)
    #expect(CosyVoiceStreamingClient.configured(environment: [
        "AI_INTERPRETER_TTS_SERVER_URL": "https://voice.example.test"
    ]) != nil)
}

@Test func cosyVoiceExplicitDisableOverridesPersistedServerURL() {
    #expect(CosyVoiceStreamingClient.configured(environment: [
        "AI_INTERPRETER_DISABLE_REMOTE_TTS": "1",
        "AI_INTERPRETER_TTS_SERVER_URL": "https://voice.example.test",
        "AI_INTERPRETER_TTS_SERVER_TOKEN": "persisted-token",
    ]) == nil)
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

@Test func cosyVoiceChunksWholePCMBodyWithoutPerByteAsyncIteration() {
    let initialBytes = CosyVoiceStreamingClient.initialFramesPerChunk
        * MemoryLayout<Int16>.size
    let steadyBytes = CosyVoiceStreamingClient.steadyFramesPerChunk
        * MemoryLayout<Int16>.size
    let body = Data(repeating: 7, count: initialBytes + steadyBytes * 2 + 400)
    let chunks = CosyVoiceStreamingClient.transportChunks(body)
    #expect(chunks.map(\.count) == [initialBytes, steadyBytes, steadyBytes, 400])
    #expect(chunks.reduce(into: Data(), { $0.append($1) }) == body)
}

@Test func cosyVoiceCanBuildAContinuityFirstTransportFoothold() {
    let initialBytes = CosyVoiceStreamingClient.continuityInitialFramesPerChunk
        * MemoryLayout<Int16>.size
    let steadyBytes = CosyVoiceStreamingClient.steadyFramesPerChunk
        * MemoryLayout<Int16>.size
    let body = Data(repeating: 5, count: initialBytes + steadyBytes + 200)
    let chunks = CosyVoiceStreamingClient.transportChunks(
        body,
        initialFramesPerChunk: CosyVoiceStreamingClient.continuityInitialFramesPerChunk
    )
    #expect(chunks.map(\.count) == [initialBytes, steadyBytes, 200])
    #expect(chunks.reduce(into: Data(), { $0.append($1) }) == body)
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

@Test func cosyVoiceUsesDedicatedCUDAServiceWhenConfigured() throws {
    let configuration = try #require(
        CUDAStreamingServerConfiguration.configured(environment: [
            "AI_INTERPRETER_STREAMING_SERVER_URL": "https://asr.example.test",
            "AI_INTERPRETER_TTS_SERVER_URL": "https://voice.example.test",
        ])
    )
    let client = CosyVoiceStreamingClient.configured(from: configuration)
    #expect(client.endpoint.absoluteString == "https://voice.example.test")
}

@Test func cosyVoiceBuildsAuthenticatedWebSocketURL() throws {
    let client = try #require(CosyVoiceStreamingClient.configured(environment: [
        "AI_INTERPRETER_TTS_SERVER_URL": "https://voice.example.test/base",
        "AI_INTERPRETER_TTS_SERVER_TOKEN": "secret token",
    ]))
    let components = try #require(URLComponents(
        url: client.webSocketURL(), resolvingAgainstBaseURL: false
    ))
    #expect(components.scheme == "wss")
    #expect(components.path == "/base/v1/tts/ws")
    #expect(components.queryItems?.first(where: { $0.name == "token" })?.value == "secret token")
}

@Test func cosyVoiceUsesShortNaturalPrimingTextForBothDirections() {
    #expect(CosyVoiceStreamingClient.primingText(for: .korean) == "준비됐습니다.")
    #expect(CosyVoiceStreamingClient.primingText(for: .english) == "Ready to begin.")
}

@Test func cosyVoicePullsNumberedPCMJobChunksThroughBufferedGateway() async throws {
    TTSJobURLProtocol.state.reset()
    let configuration = URLSessionConfiguration.ephemeral
    configuration.protocolClasses = [TTSJobURLProtocol.self]
    let session = URLSession(configuration: configuration)
    let client = CosyVoiceStreamingClient(
        endpoint: URL(string: "https://pcmjob.example.test")!,
        authorization: "secret", session: session
    )
    let output = AudioChunkBox()

    try await client.synthesize(text: "안녕하세요", language: .korean) {
        output.append($0)
    }

    #expect(TTSJobURLProtocol.state.paths
        == ["/v1/tts/jobs", "/v1/tts/jobs/job-1/chunks/0"])
    let chunks = output.chunks
    #expect(chunks.count == 1)
    #expect(chunks[0].samples.count == 2)
    #expect(chunks[0].samples[0] < 0)
    #expect(chunks[0].samples[1] > 0)
}
