import React from "react";
import { Frame } from "./100_components/100_Frame";
import { ErrorBoundary } from 'react-error-boundary'
import { library } from "@fortawesome/fontawesome-svg-core";
import { fas } from "@fortawesome/free-solid-svg-icons";
import { far } from "@fortawesome/free-regular-svg-icons";
import { fab } from "@fortawesome/free-brands-svg-icons";
import { useAppSetting } from "./003_provider/AppSettingProvider";
library.add(fas, far, fab);


const MyFallbackComponent = () => {
    console.log("FALLBACK")
    return (
        <div role="alert">
            <p>Something went wrong: clear setting and reloading...</p>
        </div>
    )
}

const App = () => {
    const { applicationSetting } = useAppSetting()
    return (
        <ErrorBoundary
            FallbackComponent={MyFallbackComponent}
            onError={(error, errorInfo) => {
                console.log(error, errorInfo)
                applicationSetting.clearSetting()
                // location.reload()
            }}
            onReset={() => {
                console.log("RESET!")
                applicationSetting.clearSetting()
            }}
        >
            <div className="application-container"><Frame /></div>

        </ErrorBoundary>
    )
};

export default App;
