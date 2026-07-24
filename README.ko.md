[简体中文](README.md) | [한국어](README.ko.md) | [日本語](README.ja.md) | [English](README.en.md)

# HanziStyleForge Fusion

Windows용 실험적 한자 글꼴 재구성 도구입니다. `target.ttf`에서 글꼴 스타일을 학습하고 `ref.otf`에서 한자 구조를 가져와 설치 가능한 TTF 글꼴을 생성합니다.

> 장시간 무인 실행을 위해 체크포인트 재개, 안전 중지, 자동 재시도를 지원합니다.

## 주요 기능

- `fonts/target.ttf`에서 전체 및 지역 글꼴 스타일을 학습합니다.
- `refs/ref.otf`의 기본 글리프가 포함하는 모든 한자를 재구성합니다.
- 중국 본토, 대만, 홍콩, 일본, 한국, 전승 자형 등 다양한 참조 글꼴을 사용할 수 있습니다.
- 대상 글꼴의 라틴 문자, 숫자, 기호, 가나, 한글 및 주요 OpenType 데이터를 가능한 한 유지합니다.
- 학습, 생성, 후보 선택, QA, 벡터화, 글꼴 빌드를 자동화합니다.

## 작동 방식

```text
target.ttf: 스타일
        +
ref.otf: 한자 구조와 범위
        ↓
Style Encoder → VQ → Diffusion → Refiner / Retrieval / IDS
        ↓
후보 선택 → QA → 윤곽선 변환 → TTF
```

프로그램은 어느 지역 자형이 더 올바른지 판단하지 않습니다. 최종 한자 구조는 `ref.otf`의 기본 Unicode `cmap` 글리프를 따릅니다.

## 요구 사항

- Windows 11 64-bit
- CUDA를 지원하는 NVIDIA GPU
- Python 3.10 이상
- 최소 150 GB의 여유 디스크 공간 권장

입력 글꼴:

```text
fonts\target.ttf
refs\ref.otf
```

정적 글꼴 사용을 권장합니다. `target.ttf`에는 TrueType `glyf` 테이블이 있어야 합니다. `ref.otf`는 정적 TrueType 또는 정적 CFF OTF를 사용할 수 있습니다. 가변 글꼴, TTC, OTC는 지원하지 않습니다.

## 빠른 시작

1. 저장소를 다운로드하거나 복제합니다.
2. 스타일 원본 글꼴을 `fonts\target.ttf`에 넣습니다.
3. 구조 참조 글꼴을 `refs\ref.otf`에 넣습니다.
4. 환경을 설치합니다.

   ```text
   install_cuda130.bat
   ```

5. 프로젝트를 확인합니다.

   ```text
   verify_project.bat
   ```

6. 전체 파이프라인을 시작하거나 재개합니다.

   ```text
   run_months_resilient.bat
   ```

상태 확인:

```text
run_status.bat
```

안전 중지 요청:

```text
request_safe_stop.bat
```

재개하기 전에 중지 표시를 지웁니다.

```text
clear_safe_stop.bat
run_months_resilient.bat
```

## 출력

주요 출력:

```text
build\target-HanziStyleForge-Fusion.ttf
build\target-HanziStyleForge-Fusion.ttf.report.json
work_hanzistyleforge_fusion_months\qa\index.html
```

학습 데이터, 체크포인트, 생성 진행 상태는 다음 폴더에 저장됩니다.

```text
work_hanzistyleforge_fusion_months\
```

학습 중에는 이 폴더를 삭제하지 마십시오.

## 사용 전 확인 사항

- 전체 실행에는 며칠, 몇 주 또는 그 이상이 걸릴 수 있습니다.
- 저장소에는 글꼴 파일, 사전 학습 가중치 또는 타사 글꼴 데이터셋이 포함되지 않습니다.
- 생성 글꼴에는 `target.ttf`와 `ref.otf`의 라이선스가 모두 적용될 수 있습니다.
- 학습, 수정, 재배포 권한이 있는 글꼴만 사용하십시오.
- 이 프로젝트는 실험적입니다. 배포 전에 QA 페이지와 최종 글꼴을 직접 확인하십시오.

## 연구 및 참고 자료

HanziStyleForge Fusion은 독립 구현입니다. 다음 프로젝트와 논문은 아키텍처 설계에 참고되었습니다. 해당 프로젝트의 소스 코드, 사전 학습 가중치, 글꼴 데이터셋은 이 저장소에 포함되지 않습니다.

| 출처 | 참고한 방향 |
|---|---|
| [zi2zi](https://github.com/kaonashi-tyc/zi2zi) | 한자 스타일 변환, 내용과 스타일 분리 |
| [zi2zi-JiT](https://github.com/kaonashi-tyc/zi2zi-JiT) | 다중 참조 스타일 조건, 확산 Transformer |
| [FontDiffuser](https://github.com/yeungchenwa/FontDiffuser) | 확산 생성, 다중 스케일 내용 집계, 명시적 스타일 제약 |
| [HanziGen](https://github.com/wangwenho/HanziGen) | VQ 표현과 조건부 잠재 확산 |
| [VQ-Font](https://github.com/Yaomingshuai/VQ-Font) | 이산 글꼴 token과 구조 인식 강화 |
| [LF-Font / MX-Font](https://github.com/clovaai/fewshot-font-generation) | 지역 부품 스타일, 인자 분해, 다중 전문가 |
| [DeepVecFont-v2](https://github.com/yizhiwang96/deepvecfont-v2) | Transformer 벡터 시퀀스와 윤곽선 보정 |
| [Efficient and Scalable Chinese Vector Font Generation via Component Composition](https://arxiv.org/abs/2404.06779) | 부품 영역 변환과 대규모 조합 |
| [cjkvi/cjkvi-ids](https://github.com/cjkvi/cjkvi-ids) | Unicode IDS 부품 구조와 지역 힌트 |

인용은 방법상의 참고만 의미하며, 상위 프로젝트의 코드, 가중치, 데이터 또는 글꼴을 복사할 권한을 부여하지 않습니다. 타사 자료를 사용하기 전에 현재 라이선스와 이용 약관을 확인하십시오.

## 라이선스

프로젝트 코드 라이선스는 [LICENSE](LICENSE)를 참조하십시오. 사용자가 제공한 글꼴, 생성된 글꼴 및 타사 자료에는 각각의 조건이 적용됩니다.

## 기여

Issue와 Pull Request를 환영합니다. 타사 코드, 데이터 또는 모델을 추가할 때는 출처와 라이선스 정보를 함께 명시하십시오.
