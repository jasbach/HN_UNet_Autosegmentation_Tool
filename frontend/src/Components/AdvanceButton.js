import Button from '@mui/material/Button';
import { CircularProgress } from '@mui/material';
import TextareaAutosize from '@mui/base/TextareaAutosize';
import axios from 'axios';
import React, { useState } from 'react';
import { saveAs } from 'file-saver';

function AdvanceButton() {
    const [buttonMode,setButtonMode] = useState("Validate")
    const [activeLoading, setActiveLoading] = useState(false)
    const [currentStatus, setCurrentStatus] = useState("Pending file upload and validation to begin neural network inference.")
    const [fileID, setFileID] = useState("")

    function advance() {
        var CNN_threadID = ""
        var interval_ID = 0
        if (buttonMode == "Validate") {
            setActiveLoading(true)
            axios.get("http://localhost:5000/api/files/validate")
            .then((res) => {
                setCurrentStatus(res.data)
                setActiveLoading(false)
                setButtonMode("Process")
            })
            .catch((res) => {
                setCurrentStatus(res.data)
                setActiveLoading(false)
                setButtonMode("Validate")
            })
        } else if (buttonMode == "Process") {
            setActiveLoading(true)
            axios.get('http://localhost:5000/api/threads/create?threadtype=neural')
            .then((res) => {
                CNN_threadID = res.data
                interval_ID = setInterval(progressUpdate,1000,CNN_threadID)
            }).then(() => {
                axios.get('http://localhost:5000/api/inference?thread_id=' + CNN_threadID)
                .then((res) => {
                    setFileID(res.data)
                    setActiveLoading(false)
                    clearInterval(interval_ID)
                })
            }).catch(() => {
                setCurrentStatus("An error has occurred.")
                setButtonMode("Validate")
            })
            setButtonMode("Download")
        } else if (buttonMode == "Download") {
            window.open("http://localhost:5000/api/files/download?file_id=" + fileID)
        }
    }

    function progressUpdate(CNN_threadID) {
        axios.get('http://localhost:5000/api/threads/' + CNN_threadID + "/progress")
        .then((res) => {
            setCurrentStatus(res.data)
        }).catch(() => setCurrentStatus("Error has occurred."))
    }

    return (
        <div>
            { activeLoading ? <CircularProgress /> : null}
            <br/>
            { activeLoading ? <Button variant="contained" size="small" disabled>Working...</Button> :
            <Button onClick={advance} variant="contained" size="small">
                { buttonMode }
            </Button>
            }
            <br/><br/>
            <TextareaAutosize readOnly value={ "STATUS:\n" + currentStatus } style={{width: 600}} />
            <br/>
            { buttonMode == "Download" ? <Button onClick={ () => window.location.reload(false) } color='warning' sx={{ mb: 3}}>RESET ALL</Button> : null }
        </div>
    )
}

export default AdvanceButton;