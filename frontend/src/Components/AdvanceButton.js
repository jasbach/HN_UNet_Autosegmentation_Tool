import Button from '@mui/material/Button';
import { CircularProgress } from '@mui/material';
import axios from 'axios';
import React, { useState } from 'react'

function AdvanceButton() {
    const [buttonMode,setButtonMode] = useState("Validate")
    const [activeLoading, setActiveLoading] = useState(false)

    function advance() {
        if (buttonMode == "Validate") {
            setActiveLoading(true)
            setTimeout(() => {
                setButtonMode("Process")
            }, 3000);
            setTimeout(() => {
                setActiveLoading(false)
            }, 3000);
        } else if (buttonMode == "Process") {
            setButtonMode("Download")
        } else if (buttonMode == "Download") {
            setButtonMode("Validate")
        }
    }

    return (
        <div>
            { activeLoading ? <CircularProgress /> : null}
            <br/>
            <Button onClick={advance} variant="contained" size="small">
                { buttonMode }
            </Button>
        </div>
    )
}

export default AdvanceButton;