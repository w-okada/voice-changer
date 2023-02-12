import React, { useMemo, useState } from "react";
import { useAppState } from "../../003_provider/AppStateProvider";
import { convert48KhzTo24Khz, generateTextFileName, generateWavFileName } from "../../const";
import JSZip from "jszip";
import { useAppSetting } from "../../003_provider/AppSettingProvider";

export const ExportController = () => {
    const { appStateStorageState } = useAppSetting()
    const { corpusDataState } = useAppState();
    const [_exporting, setExporting] = useState<boolean>(false);



    const updateProgress = (totalNum: number, processedNum: number) => {
        const progress = document.getElementById("export-controller-export-progress") as HTMLDivElement;
        progress.innerText = `${processedNum}/${totalNum}`;
    };

    const exportWav = async () => {
        const zip = new JSZip();

        const totalNum = Object.values(corpusDataState.corpusTextData).reduce((prev, cur) => {
            return prev + cur.text.length
        }, 0)


        let processedNum = 0;
        setExporting(true);
        updateProgress(totalNum, 0);

        for (let i = 0; i < Object.values(corpusDataState.corpusTextData).length; i++) {
            const targetCorpus = Object.values(corpusDataState.corpusTextData)[i]
            const prefix = targetCorpus.wavPrefix
            for (let j = 0; j < targetCorpus.text.length; j++) {
                processedNum++;
                updateProgress(totalNum, processedNum);

                const userData = await appStateStorageState.loadUserData("", prefix, j)
                if (!userData || !userData.micWavBlob || !userData.vfWavBlob) {
                    continue;
                }

                const fileName = generateWavFileName(prefix, j);
                const textFileName = generateTextFileName(prefix, j)
                const textData = targetCorpus.text[j]
                const textHiraData = targetCorpus.text_hira[j]
                // 生データ
                zip.file(`00_myvoice/raw/${fileName}`, userData.micWavBlob);

                // // 24Khzデータ
                // const wav24Khz = convert48KhzTo24Khz(userData.micWavBlob);
                // zip.file(`raw24k/${fileName}`, wav24Khz);
                // updateProgress(corpus.text.length * 6, processedNum++);


                // Cropデータ
                const region = userData.region
                const start = region ? region[0] : 0;
                const end = region ? region[1] : 0;
                const wav24KhzTrim = convert48KhzTo24Khz(userData.micWavBlob, start, end);
                zip.file(`00_myvoice/rawTrim24k/${fileName}`, wav24KhzTrim);

                // VF生データ
                zip.file(`00_myvoice/vf/${fileName}`, userData.vfWavBlob);

                // // VF 24Khzデータ
                // const vfWav24Khz = convert48KhzTo24Khz(userData.vfWavBlob);
                // zip.file(`vf24k/${fileName}`, vfWav24Khz);
                // updateProgress(corpus.text.length * 6, processedNum++);

                // VF Cropデータ
                const vfWav24KhzTrim = convert48KhzTo24Khz(userData.vfWavBlob, start, end);
                zip.file(`00_myvoice/wav/${fileName}`, vfWav24KhzTrim);

                // TXT
                zip.file(`00_myvoice/text/${textFileName}`, textHiraData);
                zip.file(`00_myvoice/readable_text/${textFileName}`, textData);
            }

        }

        // for (let i = 0; i < corpus.text.length; i++) {
        //     const userData = await appStateStorageState.loadUserData(title, prefix, i)
        //     if (!userData || !userData.micWavBlob || !userData.vfWavBlob) {
        //         updateProgress(corpus.text.length * 6, (processedNum += 6));
        //         continue;
        //     }

        //     const fileName = generateWavFileName(prefix, i);

        //     // 生データ
        //     zip.file(`raw/${fileName}`, userData.micWavBlob);
        //     updateProgress(corpus.text.length * 6, processedNum++);

        //     // 24Khzデータ
        //     const wav24Khz = convert48KhzTo24Khz(userData.micWavBlob);
        //     zip.file(`raw24k/${fileName}`, wav24Khz);
        //     updateProgress(corpus.text.length * 6, processedNum++);


        //     // Cropデータ
        //     const region = userData.region
        //     const start = region ? region[0] : 0;
        //     const end = region ? region[1] : 0;
        //     // console.log("REGION", region)
        //     const wav24KhzTrim = convert48KhzTo24Khz(userData.micWavBlob, start, end);
        //     zip.file(`rawTrim24k/${fileName}`, wav24KhzTrim);
        //     updateProgress(corpus.text.length * 6, processedNum++);

        //     // VF生データ
        //     zip.file(`vf/${fileName}`, userData.vfWavBlob);
        //     updateProgress(corpus.text.length * 6, processedNum++);

        //     // VF 24Khzデータ
        //     const vfWav24Khz = convert48KhzTo24Khz(userData.vfWavBlob);
        //     zip.file(`vf24k/${fileName}`, vfWav24Khz);
        //     updateProgress(corpus.text.length * 6, processedNum++);

        //     // VF Cropデータ
        //     const vfWav24KhzTrim = convert48KhzTo24Khz(userData.vfWavBlob, start, end);
        //     zip.file(`vfTrim24k/${fileName}`, vfWav24KhzTrim);
        //     updateProgress(corpus.text.length * 6, processedNum++);

        // }

        setExporting(false);
        const blob = await zip.generateAsync({ type: "blob" });
        const url = URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = `myvoice.zip`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    };

    const exportButton = useMemo(() => {
        const className = "button";

        return (
            <>
                <div className={className} onClick={exportWav}>
                    Export
                </div>
            </>
        );
    }, [corpusDataState]);

    return (

        <>
            {exportButton}
            <span id="export-controller-export-progress" className="text" ></span >

        </>

    );
};
