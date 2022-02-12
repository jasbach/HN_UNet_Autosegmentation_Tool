import React, { useState } from 'react'
import { Button, IconButton } from '@mui/material';
import RefreshIcon from '@mui/icons-material/Refresh';
import Input from '@mui/material/Input';
import List from '@mui/material/List';
import axios from 'axios'

function UploadForm() {

    const [DCMfiles,setDCMfiles] = useState([])

    function updateFileCount() {
        axios.get("http://localhost:5000/api/files",{
        headers: {"accepts":"application/json"}
        }).then(res => {setDCMfiles(res.data.files)      
        })
    }

    function cleanUp() {
        axios.delete("http://localhost:5000/api/cleanup", {
        'Content-Type': 'application/json'
        }
        )
        .then((res) => {
        alert("Files cleared successfully")
        }).catch((res) => alert("Files not cleared"))
        updateFileCount()
    }


    function handleSubmit(event) {
        const form = event.currentTarget
        var formData = new FormData();
        event.preventDefault();
        console.log(event.target.dicom_files.files)
        for (let step = 0; step < event.target.dicom_files.files.length; step++) {
            formData.append(step,event.target.dicom_files.files[step])
        }
        axios.post('http://localhost:5000/api/files',
                        formData, {
                            headers: {
                                'Content-Type': 'multipart/form-data'
                            }
                        }
        ).then((res) => {
            alert("Files successfully uploaded");
        }).catch((err) => alert("Failed to upload"))
        updateFileCount()
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
                        inputProps={{ multiple: true }} 
                    />
                </label>
                <button type="submit">
                    Upload
                </button>
            </form>
            <p>
                <Button onClick={updateFileCount} size="small">REFRESH</Button>
    
                Files staged: { DCMfiles.length }
            </p>
            <Button onClick={cleanUp} color='warning'>CLEAR FILES</Button>
        </div>
    )
}

export default UploadForm;