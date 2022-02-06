import React, { useState, useEffect, useRef } from 'react'
import Button from '@mui/material/Button';
import Input from '@mui/material/Input';

function UploadForm({onFileSelect}) {

    const [DCMfiles,setDCMfiles] = useState([])
    const fileInput = useRef(null)

    function handleChange(e) {
        console.log(e.target.files)
        setDCMfiles(e.target.files[0])
    }

    return (
        <div className="uploadform">
            <Input 
                type="file" 
                // value={DCMfiles}
                inputProps={{ multiple: true }}
                onChange={(e) => handleChange(e)} 
            />
            <br/>
        </div>
    )
}

export default UploadForm;