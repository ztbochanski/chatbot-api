import './App.css';

function App() {
  return (
    <div className="App">
      <header className="App-header">


        <div style={{ height: '100%', width: '100%' }}>
          map chat out here
        </div>
        <form>
          <div className="field has-addons">
            <div className="control is-expanded">
              <input className="input" name="userInput" type="text" placeholder="Type your message" />
            </div>
            <div className="control">
              <button className="button is-info">
                Send
              </button>
            </div>
          </div>
        </form>

      </header>
    </div>
  );
}

export default App;
