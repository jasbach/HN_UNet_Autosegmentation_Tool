import React, { useState, useEffect } from 'react'
import './App.css'
import UploadForm from './Components/UploadForm'
import AdvanceButton from './Components/AdvanceButton'
import Button from '@mui/material/Button'
import axios from 'axios'


function App(props) {

  
  

  return (
    <div className="App">
      <h1>
        Autosegmentation tool for head and neck
      </h1>
      <UploadForm />
      <p>
        <AdvanceButton />
      </p>
    </div>

  )
}



export default App