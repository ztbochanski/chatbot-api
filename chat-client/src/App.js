import React, { useEffect, useState } from 'react';
import io from 'socket.io-client';

import logo from './logo.svg'
import 'bulma/css/bulma.min.css'

const App = () => {
  const [socket, setSocket] = useState(null)
  const [value, setValue] = useState('')
  const [context, setContext] = useState('')
  const [messages, setMessages] = useState([])

  useEffect(() => {
    const newSocket = io(`http://localhost:5001`)
    setSocket(newSocket)
    return () => newSocket.close()
  }, [setSocket])

  const submitForm = (e) => {
    e.preventDefault()
    setMessages((previousMessages) => [
      ...previousMessages, value
    ])
    socket.emit('client event', {
      message: value,
      context: context
    } )
    setValue('')
  }

  useEffect(() => {
    const messageListener = (payload) => {
      setContext(payload.context)
      setMessages((previousMessages) => [
        ...previousMessages, payload.message
      ])
    }

    if (socket != null){
      socket.on('server event', messageListener)
      return () => {
        socket.off('message', messageListener)
      }
    }
  }, [socket])

  return (
    <div className="">
      
      <section className="hero is-fullheight">
      
        <div className="hero-head">
          
          <header className="hero is-primary is-bold">
            <div className="hero-body">
            
              <div className="level">
                <div className="level-item has-text-centered ml-4">
                  <div>
                    <h1 className="title">
                      Surgery Recover Messenger
                    </h1>
                  </div>
                </div>
              </div>

            </div>
          </header>

        </div>

        <div className="hero-body">
          
          <div style={{ height: '100%', width: '100%' }}>
            
            {messages.map((message, i) => {
              const msgClass = i === 0 || i % 2 === 0
              return (
                <p style={{ padding: '.25em', textAlign: msgClass ? 'left' : 'right', overflowWrap: 'normal' }}>
                  <span key={i} className={`tag is-medium ${msgClass ? 'is-success' : 'is-info'}`}>{message}</span>
                </p>
              )}
            )}

          </div>


          <div className="level-item has-text-centered">
            <div>
              <figure className="image">
                <img src={logo} className="App-logo" alt="logo" />
              </figure>
              </div>
            </div>
          </div>

        <div className="hero-foot">
          <footer className="section is-small">
            

            <form onSubmit={submitForm}>
              <div className="field has-addons">
                <div className="control is-expanded">
                  <input 
                    className="input" 
                    name="userInput" 
                    type="text"  
                    placeholder="Ask your question"
                    autoFocus
                    value={value}
                    onChange={(e) => {
                      setValue(e.currentTarget.value)
                    }}
                  />
                </div>
                <div className="control">
                  <button className="button is-primary">
                    Send
                  </button>
                </div>
              </div>
            </form>

          </footer>
        </div>
      </section>
    </div>
  )
}

export default App
