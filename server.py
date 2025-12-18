import subprocess
import threading
import time
import os
import signal
import sys
from flask import Flask, render_template_string, request, jsonify

app = Flask(__name__)

JAIDE_BINARY_PATH = './src/main'

proc = None

def start_jaide_process():
    global proc
    try:
        proc = subprocess.Popen(
            [JAIDE_BINARY_PATH],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
    except FileNotFoundError:
        print(f"Error: Binary {JAIDE_BINARY_PATH} not found.")
        sys.exit(1)
    except Exception as e:
        print(f"Error starting process: {e}")
        sys.exit(1)

start_jaide_process()

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="hu">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
    <title>JAIDE v40 Console</title>
    <style>
        * { box-sizing: border-box; }
        body { 
            background-color: #000000; 
            color: #00ff41; 
            font-family: 'Courier New', Courier, monospace; 
            margin: 0; 
            display: flex; 
            flex-direction: column; 
            height: 100vh; 
            overflow: hidden;
        }
        .header { 
            padding: 15px; 
            background: #0d0d0d; 
            border-bottom: 1px solid #00ff41; 
            text-align: center; 
            font-weight: bold; 
            letter-spacing: 2px;
            text-transform: uppercase;
            box-shadow: 0 0 10px #00ff41;
            z-index: 10;
        }
        .container {
            display: flex;
            flex: 1;
            overflow: hidden;
            position: relative;
        }
        .scanlines {
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: linear-gradient(to bottom, rgba(255,255,255,0), rgba(255,255,255,0) 50%, rgba(0,0,0,0.1) 50%, rgba(0,0,0,0.1));
            background-size: 100% 4px;
            pointer-events: none;
            z-index: 5;
        }
        .terminal-window {
            flex: 1;
            display: flex;
            flex-direction: column;
            padding: 0;
            background: #000000;
        }
        .logs { 
            flex: 1; 
            overflow-y: auto; 
            padding: 20px; 
            font-size: 14px; 
            line-height: 1.5;
            scroll-behavior: smooth;
        }
        .logs::-webkit-scrollbar {
            width: 8px;
            background: #0d0d0d;
        }
        .logs::-webkit-scrollbar-thumb {
            background: #00ff41;
        }
        .entry {
            margin-bottom: 12px;
            animation: fadeIn 0.3s ease;
            word-wrap: break-word;
        }
        .user-entry {
            color: #ffffff;
            text-align: right;
            margin-left: 20%;
            border-right: 2px solid #ffffff;
            padding-right: 10px;
        }
        .system-entry {
            color: #00ff41;
            margin-right: 20%;
            border-left: 2px solid #00ff41;
            padding-left: 10px;
            text-shadow: 0 0 5px #00ff41;
        }
        .error-entry {
            color: #ff0000;
            border-left: 2px solid #ff0000;
            padding-left: 10px;
        }
        .input-area { 
            padding: 15px; 
            background: #0d0d0d; 
            border-top: 1px solid #00ff41; 
            display: flex; 
            gap: 10px;
            z-index: 10;
        }
        input { 
            flex: 1; 
            padding: 12px; 
            border: 1px solid #00ff41; 
            background: #000000; 
            color: #00ff41; 
            font-family: 'Courier New', Courier, monospace;
            font-size: 16px; 
            outline: none;
        }
        input:focus {
            box-shadow: 0 0 10px #004411;
        }
        button { 
            padding: 12px 25px; 
            background: #003300; 
            color: #00ff41; 
            border: 1px solid #00ff41; 
            font-family: 'Courier New', Courier, monospace;
            font-weight: bold; 
            cursor: pointer;
            transition: all 0.2s;
        }
        button:active {
            background: #00ff41;
            color: #000000;
        }
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(5px); }
            to { opacity: 1; transform: translateY(0); }
        }
    </style>
</head>
<body>
    <div class="header">JAIDE v40 :: AGI ROOT TERMINAL</div>
    <div class="container">
        <div class="scanlines"></div>
        <div class="terminal-window">
            <div class="logs" id="terminal-output">
                <div class="entry system-entry">SYSTEM ONLINE. NEURO-SYMBOLIC CORE ACTIVE.<br>MODELS LOADED SUCCESSFULLY.<br>WAITING FOR INPUT...</div>
            </div>
            <div class="input-area">
                <input type="text" id="command-input" placeholder="ENTER COMMAND OR QUERY..." autocomplete="off" autofocus>
                <button onclick="executeCommand()">EXECUTE</button>
            </div>
        </div>
    </div>

    <script>
        const terminalOutput = document.getElementById('terminal-output');
        const commandInput = document.getElementById('command-input');

        commandInput.addEventListener('keypress', function (e) {
            if (e.key === 'Enter') {
                executeCommand();
            }
        });

        async function executeCommand() {
            const cmd = commandInput.value;
            if (!cmd) return;

            appendLog(cmd, 'user-entry');
            commandInput.value = '';

            try {
                const response = await fetch('/interact', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({input: cmd})
                });

                if (!response.ok) {
                    throw new Error('Network response was not ok');
                }

                const data = await response.json();
                appendLog(data.output, 'system-entry');
            } catch (e) {
                appendLog("CONNECTION ERROR: AGENT UNREACHABLE", 'error-entry');
            }
        }

        function appendLog(text, className) {
            const div = document.createElement('div');
            div.className = 'entry ' + className;
            div.innerText = text;
            terminalOutput.appendChild(div);
            terminalOutput.scrollTop = terminalOutput.scrollHeight;
        }
    </script>
</body>
</html>
"""

@app.route('/')
def home():
    return render_template_string(HTML_TEMPLATE)

@app.route('/interact', methods=['POST'])
def interact():
    user_input = request.json.get('input', '')

    if not user_input:
        return jsonify({'output': ''})

    global proc
    if proc.poll() is not None:
        start_jaide_process()
        return jsonify({'output': 'SYSTEM RESTARTED. PLEASE RETRY.'})

    try:
        proc.stdin.write(user_input + "\n")
        proc.stdin.flush()

        output_line = proc.stdout.readline()

        if not output_line:
             return jsonify({'output': 'NO RESPONSE FROM CORE'})

        return jsonify({'output': output_line.strip()})

    except Exception as e:
        return jsonify({'output': f'SYSTEM ERROR: {str(e)}'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
