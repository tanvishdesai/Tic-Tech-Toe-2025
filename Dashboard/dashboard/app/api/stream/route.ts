import { spawn, ChildProcess } from 'child_process';
import { NextRequest, NextResponse } from 'next/server';
import path from 'path';

// Flag to prevent multiple Python processes if the API is called multiple times
// In a real production scenario, you'd manage this more robustly.
let pythonProcess: ChildProcess | null = null;

export async function GET(request: NextRequest) {
    // --- Create the SSE Stream ---
    const stream = new ReadableStream({
        start(controller) {
            console.log('SSE Stream started');

            // --- Spawn the Python Backend Script ---
            // Adjust 'python' if necessary (e.g., 'python3' or full path to venv python)
            // Set cwd to the project root so the script can find 'src/...'
            const scriptPath = path.resolve(process.cwd(), '../../streaming_backend.py'); // Navigate up from app/api/stream
            const pythonExecutable = 'python'; // Or 'python3', or './backend_venv/Scripts/python.exe' etc.
            const projectRoot = path.resolve(process.cwd(), '../../'); // Root of Tic-Tech-Toe

            console.log(`Spawning Python script: ${pythonExecutable} ${scriptPath} in ${projectRoot}`);

            // Ensure only one process runs (simple check for hackathon)
            if (pythonProcess && !pythonProcess.killed) {
                 console.warn("Python process already running. Killing existing one.");
                 pythonProcess.kill();
            }


            try {
                pythonProcess = spawn(pythonExecutable, [scriptPath], {
                    cwd: projectRoot, // Set CWD to project root
                    stdio: ['ignore', 'pipe', 'pipe'], // Ignore stdin, pipe stdout/stderr
                    shell: false, // More secure, avoids shell interpretation
                });
            } catch (error) {
                 console.error("Error spawning Python process:", error);
                 controller.enqueue(`event: error\ndata: ${JSON.stringify({ message: "Failed to start backend process." })}\n\n`);
                 controller.close();
                 return;
            }


            let buffer = '';
            const startDelimiter = '---JSON_START---';
            const endDelimiter = '---JSON_END---';

            // --- Handle Python Script Output (stdout) ---
            if (pythonProcess?.stdout) {
                pythonProcess.stdout.on('data', (data) => {
                    buffer += data.toString();
                    // console.log("Raw stdout chunk:", data.toString().substring(0, 100) + "..."); // Debug: Log received chunks

                    let startIndex = buffer.indexOf(startDelimiter);
                    let endIndex = buffer.indexOf(endDelimiter);

                    while (startIndex !== -1 && endIndex !== -1 && startIndex < endIndex) {
                        // Extract the JSON string between delimiters
                        const jsonString = buffer.substring(startIndex + startDelimiter.length, endIndex).trim();

                        // Remove the processed part (including delimiters) from the buffer
                        buffer = buffer.substring(endIndex + endDelimiter.length);

                        if (jsonString) {
                            try {
                                // We don't need to parse here, just send the raw JSON string
                                // Parsing will happen on the client side
                                 // console.log("Sending JSON data:", jsonString.substring(0,100)+"..."); // Debug
                                controller.enqueue(`data: ${jsonString}\n\n`);
                            } catch (e) {
                                console.error('Error parsing JSON from Python script:', e);
                                console.error('Problematic JSON string:', jsonString);
                                 controller.enqueue(`event: error\ndata: ${JSON.stringify({ message: "Error processing backend data." })}\n\n`);
                            }
                        }

                        // Look for the next message in the remaining buffer
                        startIndex = buffer.indexOf(startDelimiter);
                        endIndex = buffer.indexOf(endDelimiter);
                    }
                });
            }

            // --- Handle Python Script Errors (stderr) ---
            if (pythonProcess?.stderr) {
              pythonProcess.stderr.on('data', (data) => {
                  console.error(`Python stderr: ${data}`);
                   // Optionally send errors to the client via SSE events
                   controller.enqueue(`event: backend_error\ndata: ${JSON.stringify({ message: data.toString() })}\n\n`);
              });
            }

            // --- Handle Python Script Exit ---
            if (pythonProcess) {
              pythonProcess.on('close', (code) => {
                  console.log(`Python process exited with code ${code}`);
                  controller.enqueue(`event: close\ndata: ${JSON.stringify({ message: `Backend process exited with code ${code}` })}\n\n`);
                  pythonProcess = null; // Reset flag
              });
            }

            // --- Handle Python Script Spawn Error ---
             if (pythonProcess) {
               pythonProcess.on('error', (err) => {
                   console.error('Failed to start Python process:', err);
                    controller.enqueue(`event: error\ndata: ${JSON.stringify({ message: "Failed to start backend process." })}\n\n`);
                   pythonProcess = null; // Reset flag
               });
             }

            // --- Handle Client Disconnection ---
            request.signal.addEventListener('abort', () => {
                console.log('Client disconnected, killing Python process.');
                if (pythonProcess && !pythonProcess.killed) {
                    pythonProcess.kill();
                    pythonProcess = null;
                }
                controller.close();
            });

        },
        cancel() {
            console.log('SSE Stream cancelled, killing Python process.');
             if (pythonProcess && !pythonProcess.killed) {
                 pythonProcess.kill();
                 pythonProcess = null;
             }
        },
    });

    // --- Return the SSE Response ---
    return new NextResponse(stream, {
        headers: {
            'Content-Type': 'text/event-stream',
            'Cache-Control': 'no-cache',
            'Connection': 'keep-alive',
        },
    });
}
