import fs from "node:fs/promises";
import { Presentation, PresentationFile } from "@oai/artifact-tool";
import { buildSlide01 } from "./layouts/slide-01.mjs";
import { buildSlide08 } from "./layouts/slide-08.mjs";
import { buildSlide09 } from "./layouts/slide-09.mjs";
import { buildSlide11 } from "./layouts/slide-11.mjs";
import { buildSlide13 } from "./layouts/slide-13.mjs";
import { buildSlide17 } from "./layouts/slide-17.mjs";
import { buildSlide18 } from "./layouts/slide-18.mjs";
import { buildSlide26 } from "./layouts/slide-26.mjs";

const ROOT = "/Users/shinmireu/Desktop/CAU/Project/[DSIL]Highway Star";
const TMP = `${ROOT}/.codex_tmp/smart_dof_presentation`;
const ASSETS = `${TMP}/assets`;
const FINAL = `${ROOT}/Python_Workspace/Smart DOF/Smart_DoF_개발진행_발표자료.pptx`;

async function imageBytes(name) {
  const data = await fs.readFile(`${ASSETS}/${name}`);
  return data.buffer.slice(data.byteOffset, data.byteOffset + data.byteLength);
}

function addImage(slide, name, position, alt, fit = "cover") {
  return imageBytes(name).then((blob) => slide.images.add({
    blob,
    contentType: "image/png",
    alt,
    fit,
    geometry: "roundRect",
    borderRadius: "rounded-lg",
    position,
  }));
}

function addText(slide, text, position, style = {}) {
  const box = slide.shapes.add({
    geometry: "textbox",
    position,
    fill: "none",
    line: { style: "solid", fill: "none", width: 0 },
  });
  box.text = text;
  box.text.style = {
    fontSize: style.fontSize ?? 22,
    bold: style.bold ?? false,
    color: style.color ?? "#000000",
    alignment: style.alignment ?? "left",
    verticalAlignment: style.verticalAlignment ?? "top",
  };
  return box;
}

function addImageLabel(slide, text, left, top, width = 190) {
  const bg = slide.shapes.add({
    geometry: "rect",
    position: { left, top, width, height: 38 },
    fill: "#000000",
    line: { style: "solid", fill: "none", width: 0 },
  });
  const label = addText(slide, text, { left: left + 12, top: top + 5, width: width - 20, height: 28 }, {
    fontSize: 18, bold: true, color: "#FFFFFF",
  });
  return { bg, label };
}

function addTitleOverride(slide, text) {
  slide.shapes.add({
    geometry: "rect",
    position: { left: 0, top: 0, width: 1280, height: 138 },
    fill: "#FFFFFF",
    line: { style: "solid", fill: "none", width: 0 },
  });
  addText(slide, text, { left: 41, top: 36, width: 1198, height: 70 }, {
    fontSize: 39, bold: false,
  });
}

function setNotes(slide, text, sources = []) {
  const sourceBlock = sources.length
    ? `\n\n[Sources]\n${sources.map((item) => `- ${item}`).join("\n")}`
    : "";
  slide.speakerNotes.textFrame.setText(`${text}${sourceBlock}`);
  slide.speakerNotes.setVisible(true);
}

const presentation = Presentation.create({ slideSize: { width: 1280, height: 720 } });

// 1. Cover
{
  const slide = buildSlide01(presentation, {
    title: "SMART DOF",
    title2: "객체·깊이 인지형\n시네마틱 포커스 렌더링",
    title3: "뎁스맵 이후 개발 진행 및 결과 · 2026.07.27",
  });
  setNotes(slide,
    "안녕하세요. 이번 발표에서는 이전에 공유드린 뎁스맵 영상 생성과 원본 비교 이후, 그 깊이 정보를 실제 영상 연출에 어떻게 활용했는지 말씀드리겠습니다. 목표는 사용자가 지정한 객체는 선명하게 유지하고, 주변은 거리 차이에 따라 자연스럽게 흐리며, 객체가 사라질 때에는 초점이 부드럽게 풀리는 시네마틱 DoF를 만드는 것이었습니다."
  );
}

// 2. Previous baseline
{
  const slide = buildSlide11(presentation, {
    footer1: "2",
    title: "이전 발표에서는 영상의 상대 깊이를 시각화했습니다",
    body1: {
      topic: "출발점",
      loremIpsumDolorSitAmetConsecteturAdipiscing: "Depth Anything을 이용해 주행 영상의 프레임별 상대 깊이를 추정했습니다.",
      loremIpsumDolorSitAmetConsecteturAdipiscing2: "가까운 보닛·도로와 먼 건물·하늘이 서로 다른 값으로 분리되는 것을 확인했습니다.",
    },
    body2: "원본 영상",
    body3: "뎁스맵 영상",
    body4: { detailGoesHere: "RGB 장면", detailGoesHere2: "854 × 480", detailGoesHere3: "주행 영상 입력" },
    body5: { detailGoesHere: "상대 깊이", detailGoesHere2: "프레임별 추정", detailGoesHere3: "DoF 계산 기반" },
  });
  await addImage(slide, "source_45.png", { left: 42, top: 327, width: 581, height: 180 }, "원본 주행 영상 대표 프레임");
  await addImage(slide, "depth_45.png", { left: 657, top: 327, width: 581, height: 180 }, "Depth Anything 상대 깊이맵 대표 프레임");
  addImageLabel(slide, "ORIGINAL", 56, 341, 150);
  addImageLabel(slide, "DEPTH MAP", 671, 341, 170);
  setNotes(slide,
    "이전 발표의 마지막 결과입니다. 왼쪽은 원본 영상이고 오른쪽은 같은 시점의 뎁스맵입니다. 보닛과 가까운 도로는 밝게, 원거리 건물과 하늘은 어둡게 표현됩니다. 여기까지는 장면의 거리 관계를 알아낸 단계였습니다. 이번 개발에서는 이 깊이 정보를 이용해 사용자가 원하는 객체에 초점을 맞추는 영상 효과로 확장했습니다.",
    ["Local project frame: 0720 Pitch/sample_38_short.mp4", "Local project frame: 0720 Pitch/depth_output.mp4"]
  );
}

// 3. Goal
{
  const slide = buildSlide08(presentation, {
    footer1: "3",
    title: "깊이 정보에 ‘사용자의 초점 의도’를 결합했습니다",
    body1: {
      titleHere: "목표\n",
      loremIpsumDolorSitAmetConsecteturAdipiscing:
        "① 롯데마트 건물을 초점 대상으로 선택\n\n② 선택 객체는 안정적으로 선명하게 유지\n\n③ 주변은 깊이·거리 차이에 따라 점진적으로 흐림\n\n④ 대상이 사라지면 약 1.5초 동안 Deep Focus로 복귀",
    },
  });
  await addImage(slide, "v10_45.png", { left: 658, top: 145, width: 581, height: 403 }, "롯데마트를 초점 대상으로 렌더링한 V10 프레임");
  addImageLabel(slide, "TARGET: LOTTE MART", 678, 165, 245);
  addText(slide, "객체를 단순히 잘라내는 효과가 아니라,\n시선의 중심을 설계하는 후처리", { left: 680, top: 565, width: 520, height: 70 }, {
    fontSize: 23, bold: true,
  });
  addTitleOverride(slide, "깊이 정보에 사용자의 초점 의도를 더했습니다");
  setNotes(slide,
    "새로운 목표는 깊이맵 전체를 보여주는 것이 아니라 사용자가 원하는 대상을 초점의 중심으로 만드는 것이었습니다. 이번 영상에서는 롯데마트 건물을 선택했습니다. 건물은 선명하게 유지하고, 주변은 건물과의 깊이 및 공간적 거리에 따라 흐리게 했습니다. 그리고 건물이 화면에서 사라질 때 효과가 갑자기 꺼지지 않고 약 1.5초 동안 자연스럽게 전체 초점으로 돌아가게 했습니다.",
    ["Local project frame: Python_Workspace/Smart DOF/step14_v10_progressive_focus.mp4"]
  );
}

// 4. Pipeline
{
  const slide = buildSlide18(presentation, {
    footer1: "4",
    title: "인식·안정화·렌더링의 세 단계로 구성했습니다",
    body1: {
      titleHere: "1. 장면 인식\n",
      loremIpsumDolorSitAmetConsecteturAdipiscing:
        "Depth Anything\n상대 깊이 추정\n\nMobileSAM\n선택 객체 마스크 생성",
    },
    body2: {
      titleHere: "2. 시간 안정화\n",
      loremIpsumDolorSitAmetConsecteturAdipiscing:
        "Optical Flow\n객체 이동 추적\n\n전후 검증·EMA\n깜빡임과 오추적 억제",
    },
    body3: {
      titleHere: "3. 시네마틱 렌더링\n",
      loremIpsumDolorSitAmetConsecteturAdipiscing:
        "상대 깊이 기반 블러맵\n객체 경계 feathering\n\n다단 가변 블러\nCosine Focus Pull",
    },
    label1: "PERCEPTION",
    label2: "STABILIZATION",
    label3: "RENDERING",
  });
  setNotes(slide,
    "시스템은 크게 세 단계입니다. 첫째, Depth Anything으로 장면의 상대 깊이를 추정하고 MobileSAM으로 사용자가 선택한 객체의 마스크를 만듭니다. 둘째, Optical Flow로 객체의 움직임을 추적하고 전진·역방향 검증과 EMA를 사용해 프레임 간 흔들림을 줄입니다. 셋째, 깊이 차이와 객체 경계 거리를 결합해 블러맵을 만들고, 여러 블러 반경을 픽셀별로 보간해 최종 영상을 렌더링합니다."
  );
}

// 5. Iterations
{
  const slide = buildSlide17(presentation, {
    footer1: "5",
    title: "버전 반복을 통해 ‘강한 필터’에서 ‘연속적인 초점’으로 개선했습니다",
    label1: "V6",
    label2: "V8–V9",
    label3: "V10",
    body1: {
      titleHere: "객체 분리\n",
      loremIpsumDolorSitAmetConsecteturAdipiscing: "SAM 기반 초점 보호\n경계가 단단하고 블러가 과도함",
    },
    body2: {
      titleHere: "자연스러운 전환\n",
      loremIpsumDolorSitAmetConsecteturAdipiscing: "영역 프롬프트·연속 블러\n대상 유실 후 Focus Pull",
    },
    body3: {
      titleHere: "점진적 블러\n",
      loremIpsumDolorSitAmetConsecteturAdipiscing: "1px 단위 블러 반경\n넓고 세밀한 강도 분포",
    },
  });
  setNotes(slide,
    "개선은 한 번에 이루어지지 않았습니다. V6에서는 SAM을 이용해 객체를 보호했지만 객체만 도려낸 듯한 경계와 강한 배경 블러가 나타났습니다. V8과 V9에서는 건물 전체를 선택할 수 있도록 영역 프롬프트를 추가하고, 블러 신호를 부드럽게 결합했으며, 대상 이탈 후 Focus Pull을 구현했습니다. V10에서는 블러 강도가 1에서 10까지 더 촘촘하게 증가하도록 곡선과 렌더 반경을 다시 설계했습니다."
  );
}

// 6. Technical changes
{
  const slide = buildSlide13(presentation, {
    footer1: "6",
    title: "자연스러움을 결정한 네 가지 핵심 개선",
    body1: {
      titleGoesHere: "객체 전체 선택\n",
      loremIpsumDolorSitAmetConsecteturAdipiscing: "점·박스·긍정/부정 포인트를 함께 사용해 건물 전체를 초점 대상으로 정의",
    },
    body2: {
      titleGoesHere: "부드러운 경계\n",
      loremIpsumDolorSitAmetConsecteturAdipiscing: "거리 변환과 넓은 feathering으로 선명 영역과 블러 영역을 연속적으로 연결",
    },
    body3: {
      titleGoesHere: "점진적 강도\n",
      loremIpsumDolorSitAmetConsecteturAdipiscing: "중간 강도가 압축되지 않도록 깊이·공간 거리 블러를 긴 구간에 재분배",
    },
    body4: {
      titleGoesHere: "자연스러운 해제\n",
      loremIpsumDolorSitAmetConsecteturAdipiscing: "마지막 안정 마스크와 초점 깊이를 기준으로 Cosine easing 적용",
    },
  });
  setNotes(slide,
    "자연스러움을 개선한 핵심은 네 가지입니다. 먼저 단일 점이 아니라 박스와 긍정·부정 포인트를 함께 사용해 롯데마트 건물 전체를 선택했습니다. 다음으로 거리 기반 feathering을 넓혀 경계를 부드럽게 했습니다. 세 번째로 깊이 차이와 객체 거리에 따른 블러를 긴 구간에 나누어 1, 2, 3처럼 점진적으로 증가하도록 했습니다. 마지막으로 대상이 사라질 때 마지막 안정 마스크와 초점 깊이를 기준으로 cosine easing을 적용했습니다."
  );
}

// 7. Visual result
{
  const slide = buildSlide08(presentation, {
    footer1: "7",
    title: "V10에서는 중간 거리의 블러가 더 세밀하게 분포합니다",
    body1: {
      titleHere: "V9 → V10\n",
      loremIpsumDolorSitAmetConsecteturAdipiscing:
        "V9는 초점 대상과 배경의 분리는 명확했지만 중간 블러 강도가 빠르게 증가했습니다.\n\nV10은 블러 곡선을 넓히고 1px 단위 반경을 보간해 도로·차량·원경 사이의 변화가 더 점진적으로 이어집니다.",
    },
  });
  await addImage(slide, "v9v10_45.png", { left: 42, top: 190, width: 1196, height: 336 }, "V9와 V10의 동일 프레임 비교");
  addText(slide, "핵심 변화: ‘선명 / 흐림’의 이분법에서 연속적인 초점 깊이 표현으로", { left: 140, top: 557, width: 1000, height: 54 }, {
    fontSize: 25, bold: true, alignment: "center",
  });
  addTitleOverride(slide, "V10은 블러 강도를 더 촘촘하게 분배합니다");
  setNotes(slide,
    "왼쪽이 V9, 오른쪽이 V10입니다. 두 영상 모두 롯데마트를 선명하게 유지하지만 V10에서는 타깃 근처, 도로 중앙, 전경, 먼 배경의 블러 강도가 더 촘촘하게 분배됩니다. 이번 단계의 핵심은 단순히 블러를 약하게 만든 것이 아니라, 선명 영역에서 최대 블러까지 도달하는 중간 값을 더 많이 확보한 것입니다.",
    ["Local comparison frame: Python_Workspace/Smart DOF/phase14_v9_v10_progressive_focus_comparison.mp4"]
  );
}

// 8. Focus pull
{
  const slide = buildSlide08(presentation, {
    footer1: "8",
    title: "대상이 사라지면 효과가 꺼지지 않고 초점이 풀립니다",
    body1: {
      titleHere: "Focus Pull\n",
      loremIpsumDolorSitAmetConsecteturAdipiscing:
        "추적 신뢰도 저하 감지\n\n마지막 안정 객체 마스크와 초점 깊이 유지\n\n현재 프레임의 깊이로 블러맵 재계산\n\n약 1.5초 동안 블러 강도를 1 → 0으로 감쇠",
    },
  });
  await addImage(slide, "v10_125.png", { left: 658, top: 145, width: 581, height: 403 }, "타깃 이탈 후 포커스 풀링이 진행 중인 V10 프레임");
  addImageLabel(slide, "FOCUS PULL", 678, 165, 180);
  addText(slide, "TRACKED → TRANSITION → DEEP FOCUS", { left: 682, top: 572, width: 520, height: 46 }, {
    fontSize: 21, bold: true, alignment: "center",
  });
  addTitleOverride(slide, "대상 이탈 후 1.5초 동안 초점을 자연스럽게 풉니다");
  setNotes(slide,
    "기존 버전에서는 객체가 사라지는 순간 블러가 갑자기 해제됐습니다. 현재는 Optical Flow의 신뢰도가 떨어지면 잘못된 마스크를 새로 만들지 않고 전환 상태로 들어갑니다. 마지막 안정 객체 마스크와 초점 깊이를 잠시 유지하면서 현재 프레임의 깊이로 블러맵을 다시 계산하고, 약 1.5초 동안 강도를 1에서 0으로 줄여 Deep Focus로 복귀합니다.",
    ["Local project frame: Python_Workspace/Smart DOF/step14_v10_progressive_focus.mp4"]
  );
}

// 9. Status and next
{
  const slide = buildSlide09(presentation, {
    footer1: "9",
    title: "코어 기능은 갖추었고, 다음 연구 질문이 명확해졌습니다",
    body1: {
      topic: "현재 확보한 기능",
      loremIpsumDolorSitAmetConsecteturAdipiscing: "객체 선택, 객체 추적, 깊이 기반 공간가변 블러, 점진적 강도, 대상 이탈 후 Focus Pull까지 하나의 파이프라인으로 연결했습니다.",
      loremIpsumDolorSitAmetConsecteturAdipiscing2: "",
    },
    body2: { titleHere: "현재 한계", loremIpsumDolorSitAmetConsecteturAdipiscing: "단안 깊이의 흔들림\n가림 경계의 halo\n장면별 수동 프롬프트" },
    body3: { titleHere: "서비스 확장", loremIpsumDolorSitAmetConsecteturAdipiscing: "Focus Director\n복수 객체 전환\nDoF 프리셋·타임라인" },
    body4: { titleHere: "연구 확장", loremIpsumDolorSitAmetConsecteturAdipiscing: "3D Scene-Anchored\nFocal Plane\n시간 일관적 렌더링" },
  });
  setNotes(slide,
    "현재 코어 기능은 하나의 파이프라인으로 연결됐습니다. 다만 단안 깊이의 프레임 간 흔들림, 가림 경계의 halo, 장면별 프롬프트 설정은 아직 개선 과제입니다. 서비스 관점에서는 Focus Director를 통해 여러 객체의 초점 전환과 DoF 강도를 타임라인으로 편집하는 방향을 생각하고 있습니다. 연구 관점에서는 화면 좌표가 아니라 3차원 장면에 초점면을 고정하는 Scene-Anchored Focal Plane으로 확장할 계획입니다."
  );
}

// 10. Closing
{
  const slide = buildSlide11(presentation, {
    footer1: "10",
    title: "2D 블러 띠는 실제 장면의 방향과 움직임을 이해하지 못합니다",
    body1: {
      topic: "새로운 연구 문제",
      loremIpsumDolorSitAmetConsecteturAdipiscing: "일반적인 디지털 Tilt 효과는 화면 중심선과의 거리로 블러를 정합니다. 따라서 도로가 기울어지거나 카메라가 이동하면 초점 영역이 장면이 아니라 화면에 붙어 보입니다.",
      loremIpsumDolorSitAmetConsecteturAdipiscing2: "",
    },
    body2: "2D Screen-Anchored",
    body3: "3D Scene-Anchored",
    body4: {
      detailGoesHere: "같은 도로면도 서로 다른 블러",
      detailGoesHere2: "카메라 이동 시 초점 띠 고정",
      detailGoesHere3: "장면 기하와 불일치",
    },
    body5: {
      detailGoesHere: "도로·책상·벽면 방향을 추종",
      detailGoesHere2: "카메라 이동 중 초점면 유지",
      detailGoesHere3: "3D 거리로 블러 계산",
    },
  });
  setNotes(slide,
    "여기서 다음 연구 문제로 넘어갑니다. 일반적인 디지털 Tilt-Shift 효과는 화면 위에 선명한 띠를 하나 그린 뒤 그 선에서 멀어질수록 블러를 증가시킵니다. 하지만 도로는 실제로 하나의 3차원 평면입니다. 화면 좌표만 사용하면 같은 도로 위의 점들이 서로 다른 블러를 받고, 카메라가 움직여도 초점 띠는 모니터 화면에 붙어 있는 것처럼 보입니다. 제안 방향은 화면이 아니라 실제 장면의 3차원 면에 초점을 고정하는 것입니다."
  );
}

// 11. Tilted focal plane concept
{
  const slide = buildSlide08(presentation, {
    footer1: "11",
    title: "Tilt 렌즈는 초점면을 카메라와 평행하지 않게 만듭니다",
    body1: {
      titleHere: "핵심 원리\n",
      loremIpsumDolorSitAmetConsecteturAdipiscing:
        "일반 렌즈\n카메라와 평행한 초점면\n\nTilt 렌즈\n렌즈면을 기울여 초점면을 회전\n\n결과\n도로·책상처럼 기울어진 실제 면을 따라 선명도 유지\n\n※ 현재 연구 범위는 Tilt이며, Shift에 의한 프레이밍 이동은 제외",
    },
  });

  // Diagram connectors first, so labels remain legible above the lines.
  slide.shapes.add({
    geometry: "straightConnector1",
    position: { left: 705, top: 180, width: 425, height: 380 },
    fill: "none",
    line: { style: "solid", fill: "#3D8DFF", width: 5 },
  });
  slide.shapes.add({
    geometry: "straightConnector1",
    position: { left: 965, top: 260, width: 165, height: 300 },
    fill: "none",
    line: { style: "solid", fill: "#000000", width: 5 },
  });
  slide.shapes.add({
    geometry: "straightConnector1",
    position: { left: 1130, top: 205, width: 0, height: 355 },
    fill: "none",
    line: { style: "solid", fill: "#000000", width: 5 },
  });
  slide.shapes.add({
    geometry: "straightConnector1",
    position: { left: 680, top: 560, width: 500, height: 0 },
    fill: "none",
    line: { style: "solid", fill: "#B8BCC4", width: 2 },
  });
  addText(slide, "기울어진 초점면", { left: 690, top: 150, width: 230, height: 40 }, {
    fontSize: 22, bold: true, color: "#3D8DFF",
  });
  addText(slide, "렌즈면", { left: 925, top: 225, width: 120, height: 40 }, {
    fontSize: 22, bold: true,
  });
  addText(slide, "이미지면", { left: 1120, top: 160, width: 115, height: 40 }, {
    fontSize: 22, bold: true,
  });
  addText(slide, "세 면의 연장선이 만나는 관계를 이용해\n장면의 초점 방향을 회전", { left: 725, top: 585, width: 455, height: 65 }, {
    fontSize: 20, alignment: "center",
  });
  addTitleOverride(slide, "Tilt 렌즈는 기울어진 3D 초점면을 만듭니다");
  setNotes(slide,
    "실제 Tilt 렌즈에서는 렌즈면을 기울여 초점면을 카메라와 평행하지 않게 회전시킬 수 있습니다. 이미지면, 렌즈면, 초점면의 기하학적 관계는 Scheimpflug 원리로 설명됩니다. 이를 이용하면 도로나 책상처럼 카메라에서 멀어지는 면을 따라 선명도를 유지할 수 있습니다. 여기서 Tilt는 초점면을 회전하는 기능이고 Shift는 프레이밍과 원근을 조절하는 기능입니다. 이번 연구는 우선 Tilt, 즉 기울어진 초점면의 후처리 렌더링에 집중합니다.",
    ["A Versatile Camera Model for Cameras with Tilt Lenses, IJCV: https://link.springer.com/article/10.1007/s11263-016-0964-8"]
  );
}

// 12. Geometry-aware pipeline
{
  const slide = buildSlide18(presentation, {
    footer1: "12",
    title: "일반 영상에 가상의 3D 초점면을 삽입합니다",
    body1: {
      titleHere: "1. 장면 복원\n",
      loremIpsumDolorSitAmetConsecteturAdipiscing:
        "단안 깊이 추정\n카메라 움직임 추정\n\n프레임 간 깊이\nscale·shift 정렬",
    },
    body2: {
      titleHere: "2. 초점면 설정\n",
      loremIpsumDolorSitAmetConsecteturAdipiscing:
        "사용자가 같은 면의\n세 점을 선택\n\n주변 3D 점으로\nRANSAC 평면 피팅",
    },
    body3: {
      titleHere: "3. 렌더링\n",
      loremIpsumDolorSitAmetConsecteturAdipiscing:
        "광선–평면 교차로\n픽셀별 초점 거리 계산\n\nCoC·깊이 레이어 기반\n가림 인지 합성",
    },
    label1: "DEPTH + POSE",
    label2: "3D FOCAL PLANE",
    label3: "CoC RENDERING",
  });
  setNotes(slide,
    "구현 흐름은 세 단계입니다. 먼저 Depth Anything 계열 모델로 깊이를 추정하고 카메라 움직임을 구합니다. 상대 깊이는 프레임마다 크기가 바뀔 수 있기 때문에 scale과 shift를 정렬하고 시간적으로 안정화해야 합니다. 다음으로 사용자가 도로 위의 세 점을 선택하면, 주변 3D 점을 이용해 RANSAC으로 안정적인 평면을 계산합니다. 마지막으로 각 픽셀의 카메라 광선이 초점면과 만나는 거리를 구하고, 실제 장면 깊이와의 차이로 CoC를 계산합니다. 깊이 레이어를 먼 곳부터 합성해 경계 halo도 줄입니다.",
    ["VDPP: Video Depth Post-Processing for Speed and Scalability: https://openaccess.thecvf.com/content/CVPR2026W/ECV/papers/Yoon_VDPP_Video_Depth_Post-Processing_for_Speed_and_Scalability_CVPRW_2026_paper.pdf"]
  );
}

// 13. Research contribution
{
  const slide = buildSlide13(presentation, {
    footer1: "13",
    title: "논문은 ‘장면에 붙어 있는 초점면’을 핵심 기여로 설정합니다",
    body1: {
      titleGoesHere: "Scene-Anchored Plane\n",
      loremIpsumDolorSitAmetConsecteturAdipiscing: "화면 좌표가 아니라 월드 공간에 사용자 정의 초점면을 고정",
    },
    body2: {
      titleGoesHere: "Temporal Coherence\n",
      loremIpsumDolorSitAmetConsecteturAdipiscing: "깊이·카메라 움직임·가림 신뢰도를 결합해 CoC 펌핑 억제",
    },
    body3: {
      titleGoesHere: "Source Preservation\n",
      loremIpsumDolorSitAmetConsecteturAdipiscing: "생성 모델 없이 원본 픽셀을 깊이 순서로 블러·합성",
    },
    body4: {
      titleGoesHere: "정량 평가\n",
      loremIpsumDolorSitAmetConsecteturAdipiscing: "초점면 정합도·거리–블러 단조성·시간 안정성·경계 누출 측정",
    },
  });
  setNotes(slide,
    "논문의 기여는 Tilt 효과 자체가 아닙니다. 첫째, 사용자가 지정한 초점면을 화면이 아니라 월드 공간에 고정합니다. 둘째, 깊이와 카메라 움직임, 가림 신뢰도를 이용해 카메라 이동 중에도 CoC가 흔들리지 않도록 합니다. 셋째, 확산모델처럼 픽셀을 새로 생성하지 않고 원본 픽셀을 깊이 순서로 블러하고 합성합니다. 평가는 초점면 위 픽셀이 얼마나 선명한지, 초점면에서 멀어질수록 블러가 단조롭게 증가하는지, 프레임 간 블러가 안정적인지, 경계 누출이 얼마나 적은지를 측정할 계획입니다."
  );
}

// 14. Closing
{
  const slide = buildSlide26(presentation, {
    title: "NEXT",
    title2: "객체를 선택하는 DoF에서\n공간을 연출하는 Focus Director로",
    title3: {
      loremIpsumDetails: "Object Focus",
      loremIpsumDetails2: "Focus Timeline",
      loremIpsumDetails3: "3D Focal Plane",
    },
  });
  setNotes(slide,
    "정리하면, 뎁스맵 생성에서 출발해 객체 선택형 Smart DoF와 자연스러운 Focus Pull까지 구현했습니다. 다음 서비스에서는 Focus Director 안에 객체를 선택하는 Object Focus와 공간면을 선택하는 3D Focal Plane을 함께 제공할 수 있습니다. 논문에서는 장면에 고정된 기울어진 초점면, 시간적으로 안정적인 CoC, 가림 관계를 고려한 원본 보존 렌더링을 핵심 연구 문제로 구체화하겠습니다."
  );
}

await fs.mkdir(`${TMP}/rendered`, { recursive: true });
for (const [index, slide] of presentation.slides.items.entries()) {
  const number = String(index + 1).padStart(2, "0");
  const png = await presentation.export({ slide, format: "png", scale: 1 });
  await fs.writeFile(`${TMP}/rendered/slide-${number}.png`, new Uint8Array(await png.arrayBuffer()));
  const layout = await slide.export({ format: "layout" });
  await fs.writeFile(`${TMP}/rendered/slide-${number}.layout.json`, await layout.text());
}
const montage = await presentation.export({ format: "webp", montage: true, scale: 1 });
await fs.writeFile(`${TMP}/rendered/deck-montage.webp`, new Uint8Array(await montage.arrayBuffer()));
const pptx = await PresentationFile.exportPptx(presentation);
await pptx.save(FINAL);
console.log(FINAL);
