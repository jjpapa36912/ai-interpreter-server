import Foundation

enum CUDATranslationClientError: LocalizedError {
    case invalidResponse
    case server(Int, String)
    case emptyTranslation

    var errorDescription: String? {
        switch self {
        case .invalidResponse:
            "CUDA 번역 서버 응답이 올바르지 않습니다."
        case let .server(status, detail):
            "CUDA 번역 서버가 HTTP \(status) 오류를 반환했습니다: \(detail)"
        case .emptyTranslation:
            "CUDA 번역 서버가 빈 번역을 반환했습니다."
        }
    }
}

/// Text-only cloud boundary. Audio capture and low-latency ASR remain on the
/// Mac; only complete translation units cross the network. A failed request is
/// handled by ResidentLocalTranslationProvider's existing local fallback.
final class CUDATranslationClient: @unchecked Sendable {
    private struct RequestBody: Encodable {
        let text: String
        let sourceLanguage: String
        let targetLanguage: String
        let sessionID: String

        enum CodingKeys: String, CodingKey {
            case text
            case sourceLanguage = "source_language"
            case targetLanguage = "target_language"
            case sessionID = "session_id"
        }
    }

    private struct ResponseBody: Decodable {
        let translation: String
        let latencyMilliseconds: Double

        enum CodingKeys: String, CodingKey {
            case translation
            case latencyMilliseconds = "latency_ms"
        }
    }

    private struct ErrorBody: Decodable { let detail: String }

    let endpoint: URL
    private let authorization: String?
    private let sessionID: String
    private let session: URLSession

    init(
        endpoint: URL,
        authorization: String? = nil,
        sessionID: String = UUID().uuidString,
        session: URLSession = .shared
    ) {
        self.endpoint = endpoint
        self.authorization = authorization
        self.sessionID = sessionID
        self.session = session
    }

    static func configured(
        environment: [String: String] = ProcessInfo.processInfo.environment
    ) -> CUDATranslationClient? {
        guard environment["AI_INTERPRETER_DISABLE_REMOTE_TRANSLATION"] != "1" else {
            return nil
        }
        guard let raw = environment["AI_INTERPRETER_TRANSLATION_SERVER_URL"],
              let url = URL(string: raw),
              ["http", "https"].contains(url.scheme?.lowercased() ?? "")
        else { return nil }
        return CUDATranslationClient(
            endpoint: url,
            authorization: environment["AI_INTERPRETER_TRANSLATION_SERVER_TOKEN"]
        )
    }

    func translate(
        text: String, sourceLanguage: Language, targetLanguage: Language
    ) async throws -> (text: String, milliseconds: Double) {
        var request = URLRequest(url: endpoint.appending(path: "v1/translate"))
        request.httpMethod = "POST"
        request.timeoutInterval = 8
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        if let authorization {
            request.setValue("Bearer \(authorization)", forHTTPHeaderField: "Authorization")
        }
        request.httpBody = try JSONEncoder().encode(RequestBody(
            text: text,
            sourceLanguage: sourceLanguage.rawValue,
            targetLanguage: targetLanguage.rawValue,
            sessionID: sessionID
        ))
        var lastError: Error = CUDATranslationClientError.invalidResponse
        for attempt in 0..<4 {
            do {
                let (data, response) = try await session.data(for: request)
                guard let http = response as? HTTPURLResponse else {
                    throw CUDATranslationClientError.invalidResponse
                }
                if (200..<300).contains(http.statusCode) {
                    let decoded = try JSONDecoder().decode(ResponseBody.self, from: data)
                    let translation = decoded.translation
                        .trimmingCharacters(in: .whitespacesAndNewlines)
                    guard !translation.isEmpty else {
                        throw CUDATranslationClientError.emptyTranslation
                    }
                    return (translation, decoded.latencyMilliseconds)
                }
                let detail = (try? JSONDecoder().decode(ErrorBody.self, from: data).detail)
                    ?? String(data: data, encoding: .utf8)
                    ?? "unknown error"
                let serverError = CUDATranslationClientError.server(http.statusCode, detail)
                lastError = serverError
                guard [502, 503, 504].contains(http.statusCode) else { throw serverError }
            } catch {
                lastError = error
                if let urlError = error as? URLError {
                    guard [.badServerResponse, .networkConnectionLost, .cannotConnectToHost]
                        .contains(urlError.code) else { throw error }
                } else if case CUDATranslationClientError.server = error {
                    // Transient gateway status handled above.
                } else {
                    throw error
                }
            }
            if attempt < 3 {
                try await Task.sleep(for: .milliseconds(100 * (attempt + 1)))
            }
        }
        throw lastError
    }
}
