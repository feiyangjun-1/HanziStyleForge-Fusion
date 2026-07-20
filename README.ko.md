# HanziStyleForge Fusion 2.2

[English](README.md) | [简体中文](README.zh-CN.md) | [日本語](README.ja.md) | [한국어](README.ko.md)

Windows 중심으로 설계된 장시간 실행·체크포인트 재개형 한자 글리프 재구축 시스템입니다. **`target.ttf`에서는 서체 스타일만 학습**하고, **`ref.otf`에서는 Han 구조와 대상 문자 범위만 가져옵니다**. 참조 글꼴이 포함하는 모든 Han 코드포인트를 다시 생성한 뒤, 대상 글꼴의 비-Han 글리프와 주요 OpenType 엔지니어링 데이터가 보존되었는지 검증합니다.

> 현재 상태는 연구/엔지니어링용 Alpha입니다. 생성 글꼴을 배포하기 전에 시각 검수, 조판 응용 프로그램 테스트, 라이선스 검토가 필요합니다.

## 핵심 데이터 계약

```text
fonts/target.ttf  -> 스타일 학습 전용
refs/ref.otf      -> Han 구조와 대상 문자 범위 전용
```

학습은 `target.ttf`의 자기 재구성 샘플만 사용합니다.

```text
실제 target 글리프 -> target 스타일 제거 구조 프록시 -> 모델 -> 실제 target 글리프 정답
```

생성 단계에서만 `ref.otf`를 읽습니다.

```text
ref Han 구조 -> target 스타일 모델 -> target 스타일로 재구축된 Han 글리프
```

수동 동형자 목록, CN/비-CN 분류, 교차 글꼴 쌍 지도학습이 필요하지 않습니다. `hanzistyleforge/contract.py`는 참조 글꼴 경로가 학습에 들어가거나 대상 글꼴이 생성 구조로 사용되면 실행을 중단합니다.

## 주요 기능

- `ref.otf` 기본 Unicode `cmap`에 있는 모든 Han 코드포인트를 재구축합니다.
- 참조 글꼴은 중국 본토, 대만, 홍콩, 일본, 한국, 전승 자형 또는 기타 기준을 사용할 수 있습니다. 필요한 형태가 참조 글꼴의 기본 글리프여야 합니다.
- 여러 실제 target 글리프에서 전역 및 지역 스타일을 학습합니다.
- VQ 글리프 코드북, 잠재 확산, 결정론적 안전 기준선, 실제 지역 글리프 검색, 구성요소 잔차, 고해상도 Refiner, 위상 게이트, 윤곽 보정을 결합합니다.
- 확산 예측이 target 정답이 아니라 스타일 제거 구조 프록시에 가까워지면 자동 중단합니다.
- 학습, 생성, 글리프별 보정, QA, 글꼴 빌드를 영구 체크포인트에서 재개할 수 있습니다.
- 최종 글꼴은 `target.ttf`에서 시작하며 재구축 Han 글리프만 추가·재매핑합니다.
- 출력 전에 비-Han `cmap`, glyph ID, 윤곽, 메트릭, UVS, 레이아웃 테이블, 힌팅 관련 테이블을 검증합니다.

## 권장 환경

```text
Windows 11 64-bit
VRAM 12 GB 이상의 NVIDIA GPU
Python 3.10-3.14 64-bit
로컬 SSD
최소 150 GB 여유 공간
```

입력 조건:

- `fonts/target.ttf`: `glyf` 테이블이 있는 정적 TrueType. 가변 글꼴은 지원하지 않습니다.
- `refs/ref.otf`: 정적 TrueType TTF/OTF 또는 정적 CFF OTF.
- TTC/OTC와, 실행 시 `locl` 치환에 의존해야 원하는 지역 자형이 나타나는 참조 글꼴은 피하십시오.
- 배포 패키지에는 글꼴과 사전 학습 가중치가 포함되지 않습니다.

## Windows 11 빠른 시작

1. 짧은 로컬 경로에 압축을 풉니다.

   ```text
   C:\FontWork\HanziStyleForge-Fusion
   ```

2. 글꼴을 배치합니다.

   ```text
   fonts\target.ttf
   refs\ref.otf
   ```

3. CUDA 환경을 설치합니다.

   ```text
   install_cuda130.bat
   ```

4. 환경, 글꼴, 범위, 설정, 데이터 흐름 계약을 검증합니다.

   ```text
   verify_project.bat
   ```

5. 전체 장기 워크플로를 시작하거나 재개합니다.

   ```text
   run_months_resilient.bat
   ```

6. 상태를 읽기 전용으로 확인합니다.

   ```text
   run_status.bat
   ```

7. 다음 영구 체크포인트에서 안전 중지를 요청합니다.

   ```text
   request_safe_stop.bat
   ```

   재개하려면 `run_months_resilient.bat`를 다시 실행하십시오. 완료된 중지 요청은 자동으로 삭제됩니다.

8. QA 보고서가 생성된 뒤 다음을 실행할 수 있습니다.

   ```text
   open_qa.bat
   ```

## 유지되는 Windows 실행 파일

| 파일 | 용도 |
|---|---|
| `install_cuda130.bat` | `.venv` 생성, 의존성 설치, CUDA 검증 |
| `verify_project.bat` | 자체 테스트와 프로젝트 검증 |
| `run_months_resilient.bat` | 전체 재개 가능 워크플로 시작·재개 |
| `request_safe_stop.bat` | 다음 영구 체크포인트에서 중지 |
| `run_status.bat` | 상태를 읽기 전용으로 표시 |
| `open_qa.bat` | HTML QA 보고서 열기 |

고급 단계는 Python CLI로 실행할 수 있습니다.

```powershell
.venv\Scripts\python.exe hanzistyleforge.py --help
.venv\Scripts\python.exe hanzistyleforge.py --config config_fusion_months_12gb.json fusion-train
.venv\Scripts\python.exe hanzistyleforge.py --config config_fusion_months_12gb.json fusion-generate
```

## 운영 설정

주 설정 파일은 `config_fusion_months_12gb.json`입니다.

```json
{
  "training": {
    "workers": 4
  },
  "fusion": {
    "style_encoder": {
      "batch_size": 8
    }
  }
}
```

Style Encoder는 품질 게이트 기반 조기 종료를 지원합니다. 최소 100 epoch를 학습한 뒤, 최근 양성/음성 스타일 유사도 지표가 정상인 상태에서 24 epoch 동안 유의미한 검증 개선이 없으면 자동 종료합니다. 호환 체크포인트와 `history.csv`는 재사용됩니다.

현재 단계 체크포인트를 버릴 계획이 아니라면 실행 중간에 이미지 크기, 모델 채널, 스타일 차원, 잠재 채널, 코드북 크기를 변경하지 마십시오.

## 전체 워크플로

```text
입력과 CUDA 확인
-> target/ref 글리프 렌더링
-> 데이터 흐름 계약 검증
-> target 지역 스타일 아틀라스 구축
-> Style Encoder 학습
-> target VQ 코드북 학습
-> 결정론적 안전 기준선 학습
-> 다중 해상도 잠재 확산 학습
-> 실제 target 난이도 샘플 채굴
-> 고해상도 Refiner 학습
-> 윤곽 Transformer 학습
-> ref가 포함한 모든 Han 생성
-> 위상·스타일 후보 선택
-> 재개 가능한 글리프별 보정
-> QA
-> SDF/TrueType 벡터화
-> 최종 글꼴 빌드와 검증
```

전체 처리는 수 주에서 수 개월 걸릴 수 있습니다. 주요 단계와 생성 완료 글리프는 체크포인트로 저장됩니다. 복구 가능한 오류는 resilient 실행기가 재시도하지만, 영구 품질 보호 오류는 자동 재시도하지 않습니다.

## 범위와 비-Han 보호

기본 범위:

```json
{
  "scope": {
    "mode": "reference_han",
    "include_compatibility_ideographs": true
  }
}
```

참조 글꼴의 모든 Han 코드포인트는 생성 결과 또는 안전 폴백을 가져야 합니다. `require_complete=true`일 때 하나라도 누락되면 정식 출력을 중단합니다.

원래 target glyph ID를 보호하기 위해 새 glyph를 추가하므로 다음 제한이 있습니다.

```text
target glyph 수 + 추가 Han glyph 수 < 65,536
```

최종 빌드는 라틴, 키릴, 가나, 한글, 숫자, 문장부호, 기호, 비-Han 윤곽, 메트릭, Unicode 변형 시퀀스, GSUB/GPOS/GDEF/BASE/kern, TrueType 힌팅 관련 테이블을 검증합니다.

## 출력

```text
build\target-HanziStyleForge-Fusion.ttf
build\target-HanziStyleForge-Fusion.ttf.report.json
work_hanzistyleforge_fusion_months\generated\coverage.json
work_hanzistyleforge_fusion_months\generated\selection.csv
work_hanzistyleforge_fusion_months\refined\selection.csv
work_hanzistyleforge_fusion_months\qa\index.html
```

## 연구 및 참조 코드 출처

HanziStyleForge Fusion은 독립 구현입니다. 아래 공개 프로젝트와 논문은 설계 방향에 참고되었습니다. 이 저장소는 해당 프로젝트의 소스 코드, 사전 학습 가중치, 글꼴 데이터셋을 **포함하거나 재배포하지 않습니다**.

| 상위 프로젝트 | 참고한 방향 | 상위 라이선스/상태 |
|---|---|---|
| [zi2zi](https://github.com/kaonashi-tyc/zi2zi) | 한자 스타일 변환, 콘텐츠/스타일 조건 분리 | Apache-2.0 |
| [zi2zi-JiT](https://github.com/kaonashi-tyc/zi2zi-JiT) | 다중 참조 조건과 확산 Transformer 방향 | MIT 소프트웨어 라이선스와 상위 글꼴 산출물 부가 조항. 최신 조건을 확인하십시오 |
| [FontDiffuser](https://github.com/yeungchenwa/FontDiffuser) | 노이즈 제거 확산, 다중 스케일 콘텐츠 집계, 명시적 스타일 제약 | 이 릴리스 준비 당시 상위 저장소에서 라이선스 파일을 확인하지 못했습니다. 허가 없이 코드나 가중치를 복사하지 마십시오 |
| [HanziGen](https://github.com/wangwenho/HanziGen) | VQ-VAE와 잠재 확산 기반 글꼴 보완 워크플로 | Apache-2.0 |
| [VQ-Font](https://github.com/Yaomingshuai/VQ-Font) / [논문](https://doi.org/10.1609/aaai.v38i15.29577) | 이산 글꼴 token 사전분포와 구조 인식 강화 | 코드나 가중치를 복사하기 전에 현재 저장소 라이선스를 확인하십시오 |
| [LF-Font / MX-Font 통합 저장소](https://github.com/clovaai/fewshot-font-generation) | 지역 구성요소 스타일, 인수분해, 다중 전문가 | MIT. 일부 상위 모듈에는 별도 출처 고지가 있습니다 |
| [DeepVecFont-v2](https://github.com/yizhiwang96/deepvecfont-v2) | Transformer 기반 벡터 시퀀스와 윤곽 보정 | 코드는 MIT. 상위 글꼴 데이터셋에는 별도 비상업 제한이 있습니다 |
| [Efficient and Scalable Chinese Vector Font Generation via Component Composition](https://arxiv.org/abs/2404.06779) | 구성요소 영역 변환과 대규모 조합 | 논문 참고. 관련 코드와 데이터 조건은 별도로 검토하십시오 |
| [cjk-decomp](https://github.com/amake/cjk-decomp) | 지역 잔차 영역용 선택적 분해 힌트 | 다중 라이선스. 이 배포의 포함 데이터는 Apache-2.0 옵션 사용 |

자세한 출처는 [METHOD_REFERENCES.md](METHOD_REFERENCES.md), 제3자 재배포 고지는 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)를 참조하십시오.

공개 방법을 참고하는 것은 구현, 데이터셋, 글꼴, 모델 가중치를 복사할 권한을 의미하지 않습니다. 상위 자료를 추가할 경우 저작권 고지를 유지하고 현재 라이선스를 준수해야 합니다.

## 라이선스와 글꼴 권리

```text
Copyright 2026 feiyangjun_
```

HanziStyleForge 소스 코드와 프로젝트 문서는 별도로 표시된 제3자 자료를 제외하고 [Apache License 2.0](LICENSE)으로 제공됩니다.

이 라이선스는 사용자가 제공한 글꼴의 권리를 부여하지 않습니다. `target.ttf`와 `ref.otf`의 라이선스가 학습, 수정, 파생 글꼴 제작, 배포를 허용하는지 확인해야 합니다. 생성 글꼴과 체크포인트는 한쪽 또는 양쪽 입력 글꼴 라이선스의 적용을 받을 수 있습니다.

## 추가 문서

- [아키텍처](ARCHITECTURE.md)
- [데이터 흐름 계약](DATA_FLOW.md)
- [방법 참조](METHOD_REFERENCES.md)
- [제3자 고지](THIRD_PARTY_NOTICES.md)
- [테스트 보고서](TEST_REPORT.md)
- [기여 가이드](CONTRIBUTING.md)
- [보안 정책](SECURITY.md)
- [변경 기록](CHANGELOG.md)
