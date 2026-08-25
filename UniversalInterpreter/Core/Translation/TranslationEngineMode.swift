import Foundation

enum TranslationEngineMode: String, CaseIterable, Identifiable, Sendable {
    case economy
    case expressiveLocal
    case gpuStreaming

    var id: Self { self }

    var title: String {
        switch self {
        case .economy: "절약 모드"
        case .expressiveLocal: "말투 보존"
        case .gpuStreaming: "GPU 동시통역"
        }
    }

    var detail: String {
        switch self {
        case .economy: "로컬 음성 구간 감지 · 고정 음성 · 최소 API 비용"
        case .expressiveLocal: "로컬 번역 · 원문 속도·에너지·질문 억양을 고정 화자에 반영 (실험)"
        case .gpuStreaming: "GPU 음성 인식·확정 번역 · 양방향 1.5초 목표 (실험)"
        }
    }
}
