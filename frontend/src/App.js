import React, { useState, useEffect } from 'react'
import './App.css'
import UploadForm from './Components/UploadForm'
import AdvanceButton from './Components/AdvanceButton'
import { ThemeProvider, createTheme } from '@mui/material/styles';
import { green } from "@mui/material/colors";

const theme = createTheme({
    
        palette: {
          primary: {
            main: green[500],
          },
          secondary: {
            main: green['A400'],
          },
        },
    
})


function App(props) {

  
  

  return (
    <ThemeProvider theme={theme}>
      <div className="App">
        <h1>
          Autosegmentation tool for head and neck
        </h1>
        <UploadForm />
        <div>
          <AdvanceButton />
        </div>
        <footer>
            Application provided open-source for research purposes. Creators are not responsible for the use of or results from this application.
        </footer>
      </div>
    </ThemeProvider>  

  )
}



export default App