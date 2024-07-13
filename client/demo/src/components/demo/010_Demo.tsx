import React from "react"
import { GuiStateProvider } from "./001_GuiStateProvider";
import { Dialogs } from "./900_Dialogs";
import { ModelSlotControl } from "./b00_ModelSlotControl";
import { Dialogs2 } from "./910_Dialogs2";
import { ToastContainer } from 'react-toastify';
import 'react-toastify/dist/ReactToastify.css';

export const Demo = () => {
    return (
        <GuiStateProvider>
            <div className="main-body">
                <Dialogs2 />
                <Dialogs />
                <ModelSlotControl></ModelSlotControl>
            </div>
            <ToastContainer />
        </GuiStateProvider>
    )
}