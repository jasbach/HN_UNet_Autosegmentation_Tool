import React, { useState, useEffect } from 'react'
import { Button, IconButton, LinearProgress } from '@mui/material';
import RefreshIcon from '@mui/icons-material/Refresh';
import Input from '@mui/material/Input';
import axios from 'axios'

function UploadForm() {

    const [DCMfiles,setDCMfiles] = useState([])
    const [progress,setProgress] = useState(0)
    const [uploading,setUploading] = useState(false)

    useEffect(() => cleanUp(),[])

    function updateFileCount() {
        axios.get("http://localhost:5000/api/files",{
        headers: {"accepts":"application/json"}
        }).then(res => {setDCMfiles(res.data.files)      
        })
    }

    function cleanUp() {
        axios.delete("http://localhost:5000/api/cleanup", {
        'Content-Type': 'application/json'
        })
        .then((res) => console.log(res.data))
        .catch((res) => console.log(res.data))
        .then(updateFileCount())
    }

    function createThread() {}


    function handleSubmit(event) {
        const form = event.currentTarget
        var formData = new FormData();
        var upload_thread_id = '0'
        var interval_id = 0
        event.preventDefault();
        for (let step = 0; step < event.target.dicom_files.files.length; step++) {
            formData.append(step,event.target.dicom_files.files[step])
        }
        setUploading(true)
        axios.get('http://localhost:5000/api/threads/create?threadtype=upload')
        .then((res) => {
            upload_thread_id = res.data
            console.log(upload_thread_id)
            interval_id = setInterval(progressUpdate, 100, upload_thread_id)
        }).then(() => {
            axios.post('http://localhost:5000/api/files?thread_id=' + upload_thread_id,
                            formData, {
                                headers: {
                                    'Content-Type': 'multipart/form-data'
                                }
                            }
            ).then((res) => {
                console.log("Files successfully uploaded");
                clearInterval(interval_id)
            }).catch((err) => alert("Failed to upload"))
            .then(() => {
                setUploading(false)
                updateFileCount()
            }) 
        })
    }

    function progressUpdate(upload_thread_id) {
        axios.get('http://localhost:5000/api/threads/' + upload_thread_id + "/progress")
        .then((res) => {
            console.log(res.data)
            setProgress(parseInt(res.data))
        }).catch(() => setProgress(100))
    }

    return (
        <div className="uploadform">
            <form onSubmit={handleSubmit}>
                <label>File Upload:
                    <Input
                        id="DCMfiles" 
                        type="file" 
                        name="dicom_files"
                        // value={DCMfiles}
                        inputProps={{ multiple: true, accept: ".dcm" }} 
                    />
                </label>
                <button type="submit">
                    Upload
                </button>
            </form>
            <p>
                { uploading ? <LinearProgress variant="determinate" value={progress} /> : null}
                Files staged: { DCMfiles.length }
                <Button onClick={updateFileCount} size="small">REFRESH</Button>
            </p>
        </div>
    )
}

export default UploadForm;