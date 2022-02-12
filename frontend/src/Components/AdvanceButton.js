import Button from '@mui/material/Button';
import axios from 'axios';
import React, { useState } from 'react'

function AdvanceButton() {
    const [buttonMode,setButtonMode] = useState("Validate")

    function advance() {
        if (buttonMode == "Validate") {
            setButtonMode("Process")
        } else if (buttonMode == "Process") {
            setButtonMode("Download")
        } else if (buttonMode == "Download") {
            setButtonMode("Validate")
        }
    }

    return (
        <div>
            <Button onClick={advance} variant="contained" size="small">
                { buttonMode }
            </Button>
        </div>
    )
}

export default AdvanceButton;