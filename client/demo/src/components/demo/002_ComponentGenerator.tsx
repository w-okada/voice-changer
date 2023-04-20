import React from "react"
import { ClearSettingRow, ClearSettingRowProps } from "./components/102_ClearSettingRow"
import { Title, TitleProps } from "./components/101_Title"
import { StartButtonRow, StartButtonRowProps } from "./components/201_StartButtonRow"
import { PerformanceRow, PerformanceRowProps } from "./components/202_PerformanceRow"
import { ServerInfoRow, ServerInfoRowProps } from "./components/203_ServerInfoRow"
import { ModelUploaderRow, ModelUploaderRowProps } from "./components/301_ModelUploaderRow"
import { FrameworkRow, FrameworkRowProps } from "./components/302_FrameworkRow"
import { AudioInputRow, AudioInputRowProps } from "./components/401_AudioInputRow"
import { AudioOutputRow, AudioOutputRowProps } from "./components/402_AudioOutputRow"
import { GainControlRow, GainControlRowProps } from "./components/502_GainControlRow"
import { NoiseControlRow, NoiseControlRowProps } from "./components/501_NoiseControlRow"
import { F0DetectorRow, F0DetectorRowProps } from "./components/503_F0DetectorRow"
import { DividerRow, DividerRowProps } from "./components/990_DividerRow"
import { AnalyzerRow, AnalyzerRowProps } from "./components/510_AnalyzerRow"
import { SrcIdRow, SrcIdRowProps } from "./components/601_SrcIdRow"
import { DstIdRow, DstIdRowProps } from "./components/602_DstIdRow"
import { EditSpeakerIdMappingRow, EditSpeakerIdMappingRowProps } from "./components/603_EditSpeakerIdMappingRow"
import { F0FactorRow, F0FactorRowProps } from "./components/604_F0FactorRow"
import { TuneRow, TuneRowProps } from "./components/605_TuneRow"
import { ClusterInferRatioRow, ClusterInferRatioRowProps } from "./components/606_ClusterInferRatioRow"
import { NoiseScaleRow, NoiseScaleRowProps } from "./components/607_NoiseScaleRow"
import { SilentThresholdRow, SilentThresholdRowProps } from "./components/608_SilentThresholdRow"
import { InputChunkNumRow, InputChunkNumRowProps } from "./components/701_InputChunkNumRow"
import { ExtraDataLengthRow, ExtraDataLengthRowProps } from "./components/702_ExtraDataLengthRow"
import { GPURow, GPURowProps } from "./components/703_GPURow"
import { ServerURLRow, ServerURLRowProps } from "./components/801_ServerURLRow"
import { ProtocolRow, ProtocolRowProps } from "./components/802_ProtocolRow"
import { SampleRateRow, SampleRateRowProps } from "./components/803_SampleRateRow"
import { SendingSampleRateRow, SendingSampleRateRowProps } from "./components/804_SendingSampleRateRow"
import { CrossFadeOverlapSizeRow, CrossFadeOverlapSizeRowProps } from "./components/805_CrossFadeOverlapSizeRow"
import { CrossFadeOffsetRateRow, CrossFadeOffsetRateRowProps } from "./components/806_CrossFadeOffsetRateRow"
import { CrossFadeEndRateRow, CrossFadeEndRateRowProps } from "./components/807_CrossFadeEndRateRow"
import { DownSamplingModeRow, DownSamplingModeRowProps } from "./components/808_DownSamplingModeRow"
import { TrancateNumTresholdRow, TrancateNumTresholdRowProps } from "./components/809_TrancateNumTresholdRow"
import { IndexRatioRow, IndexRatioRowProps } from "./components/609_IndexRatioRow"
import { RVCQualityRow, RVCQualityRowProps } from "./components/810_RVCQualityRow"
import { ModelSamplingRateRow, ModelSamplingRateRowProps } from "./components/303_ModelSamplingRateRow"
// import { OnnxExportRow, OnnxExportRowProps } from "./components/304_OnnxExportRow"
import { EnableEnhancerRow, EnableEnhancerRowProps } from "./components/610_EnableEnhancerRow"
import { DstIdRow2, DstIdRow2Props } from "./components/602v2_DstIdRow2"
import { SilenceFrontRow, SilenceFrontRowProps } from "./components/812_SilenceFrontRow"

export const catalog: { [key: string]: (props: any) => JSX.Element } = {}

export const addToCatalog = (key: string, generator: (props: any) => JSX.Element) => {
    catalog[key] = generator
}

export const generateComponent = (key: string, props: any) => {
    if (!catalog[key]) {
        console.error("not found component generator.", key)
        return <></>
    }
    return catalog[key](props)
}

const initialize = () => {
    addToCatalog("divider", (props: DividerRowProps) => { return <DividerRow {...props} /> })

    addToCatalog("title", (props: TitleProps) => { return <Title {...props} /> })
    addToCatalog("clearSetting", (props: ClearSettingRowProps) => { return <ClearSettingRow {...props} /> })

    addToCatalog("startButton", (props: StartButtonRowProps) => { return <StartButtonRow {...props} /> })
    addToCatalog("performance", (props: PerformanceRowProps) => { return <PerformanceRow {...props} /> })
    addToCatalog("serverInfo", (props: ServerInfoRowProps) => { return <ServerInfoRow {...props} /> })

    addToCatalog("modelUploader", (props: ModelUploaderRowProps) => { return <ModelUploaderRow {...props} /> })
    addToCatalog("framework", (props: FrameworkRowProps) => { return <FrameworkRow {...props} /> })
    addToCatalog("modelSamplingRate", (props: ModelSamplingRateRowProps) => { return <ModelSamplingRateRow {...props} /> })
    // addToCatalog("onnxExport", (props: OnnxExportRowProps) => { return <OnnxExportRow {...props} /> })

    addToCatalog("audioInput", (props: AudioInputRowProps) => { return <AudioInputRow {...props} /> })
    addToCatalog("audioOutput", (props: AudioOutputRowProps) => { return <AudioOutputRow {...props} /> })

    addToCatalog("noiseControl", (props: NoiseControlRowProps) => { return <NoiseControlRow {...props} /> })
    addToCatalog("gainControl", (props: GainControlRowProps) => { return <GainControlRow {...props} /> })
    addToCatalog("f0Detector", (props: F0DetectorRowProps) => { return <F0DetectorRow {...props} /> })
    addToCatalog("analyzer", (props: AnalyzerRowProps) => { return <AnalyzerRow {...props} /> })

    addToCatalog("srcId", (props: SrcIdRowProps) => { return <SrcIdRow {...props} /> })
    addToCatalog("dstId", (props: DstIdRowProps) => { return <DstIdRow {...props} /> })
    addToCatalog("dstId2", (props: DstIdRow2Props) => { return <DstIdRow2 {...props} /> })
    addToCatalog("editSpeakerIdMapping", (props: EditSpeakerIdMappingRowProps) => { return <EditSpeakerIdMappingRow {...props} /> })
    addToCatalog("f0Factor", (props: F0FactorRowProps) => { return <F0FactorRow {...props} /> })
    addToCatalog("tune", (props: TuneRowProps) => { return <TuneRow {...props} /> })
    addToCatalog("clusterInferRatio", (props: ClusterInferRatioRowProps) => { return <ClusterInferRatioRow {...props} /> })
    addToCatalog("noiseScale", (props: NoiseScaleRowProps) => { return <NoiseScaleRow {...props} /> })
    addToCatalog("silentThreshold", (props: SilentThresholdRowProps) => { return <SilentThresholdRow {...props} /> })
    addToCatalog("indexRatio", (props: IndexRatioRowProps) => { return <IndexRatioRow {...props} /> })
    addToCatalog("enableEnhancer", (props: EnableEnhancerRowProps) => { return <EnableEnhancerRow {...props} /> })



    addToCatalog("inputChunkNum", (props: InputChunkNumRowProps) => { return <InputChunkNumRow {...props} /> })
    addToCatalog("extraDataLength", (props: ExtraDataLengthRowProps) => { return <ExtraDataLengthRow {...props} /> })
    addToCatalog("gpu", (props: GPURowProps) => { return < GPURow {...props} /> })

    addToCatalog("serverURL", (props: ServerURLRowProps) => { return <ServerURLRow {...props} /> })
    addToCatalog("protocol", (props: ProtocolRowProps) => { return <ProtocolRow {...props} /> })
    addToCatalog("sampleRate", (props: SampleRateRowProps) => { return <SampleRateRow {...props} /> })
    addToCatalog("sendingSampleRate", (props: SendingSampleRateRowProps) => { return <SendingSampleRateRow {...props} /> })
    addToCatalog("crossFadeOverlapSize", (props: CrossFadeOverlapSizeRowProps) => { return <CrossFadeOverlapSizeRow {...props} /> })
    addToCatalog("crossFadeOffsetRate", (props: CrossFadeOffsetRateRowProps) => { return <CrossFadeOffsetRateRow {...props} /> })
    addToCatalog("crossFadeEndRate", (props: CrossFadeEndRateRowProps) => { return <CrossFadeEndRateRow {...props} /> })
    addToCatalog("downSamplingMode", (props: DownSamplingModeRowProps) => { return <DownSamplingModeRow {...props} /> })
    addToCatalog("trancateNumThreshold", (props: TrancateNumTresholdRowProps) => { return <TrancateNumTresholdRow {...props} /> })
    addToCatalog("rvcQuality", (props: RVCQualityRowProps) => { return <RVCQualityRow {...props} /> })
    addToCatalog("silenceFront", (props: SilenceFrontRowProps) => { return <SilenceFrontRow {...props} /> })



}

initialize()