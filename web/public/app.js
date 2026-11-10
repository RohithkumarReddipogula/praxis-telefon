// --- Generative AI Face Visualizer ---
const canvas = document.getElementById('ai-face');
const ctx = canvas.getContext('2d');
let audioContext;
let analyser;
let dataArray;
let isSpeaking = false;

// Resize canvas for high DPI screens
function resizeCanvas() {
    const size = canvas.clientWidth;
    canvas.width = size * 2;
    canvas.height = size * 2;
}
resizeCanvas();
window.addEventListener('resize', resizeCanvas);

// Setup Audio Context (call this when the user clicks Start)
async function setupAudioVisualizer(stream) {
    audioContext = new (window.AudioContext || window.webkitAudioContext)();
    analyser = audioContext.createAnalyser();
    analyser.fftSize = 256;
    
    const source = audioContext.createMediaStreamSource(stream);
    source.connect(analyser);
    
    dataArray = new Uint8Array(analyser.frequencyBinCount);
    drawFace();
}

// Draw the face loop
function drawFace() {
    requestAnimationFrame(drawFace);
    
    const width = canvas.width;
    const height = canvas.height;
    const centerX = width / 2;
    const centerY = height / 2;
    const radius = Math.min(width, height) / 4; // Base radius

    ctx.clearRect(0, 0, width, height);

    // Get audio data if available
    let volume = 0;
    if (analyser && dataArray) {
        analyser.getByteFrequencyData(dataArray);
        // Calculate average volume
        let sum = 0;
        for (let i = 0; i < dataArray.length; i++) sum += dataArray[i];
        volume = sum / dataArray.length;
    }

    // Dynamic radius based on volume (makes the face "breathe" when speaking)
    const dynamicRadius = radius + (volume * 0.8);

    // Draw the main core (The AI Eye/Face)
    const gradient = ctx.createRadialGradient(centerX, centerY, 0, centerX, centerY, dynamicRadius);
    gradient.addColorStop(0, '#ffffff');
    gradient.addColorStop(0.5, '#00f0ff'); // Cyber Cyan
    gradient.addColorStop(1, 'rgba(112, 0, 255, 0)'); // Fade to purple/transparent

    ctx.beginPath();
    ctx.arc(centerX, centerY, dynamicRadius, 0, Math.PI * 2);
    ctx.fillStyle = gradient;
    ctx.fill();

    // Draw the inner neural network lines
    if (dataArray) {
        ctx.lineWidth = 2;
        for (let i = 0; i < dataArray.length; i += 4) {
            const angle = (i / dataArray.length) * Math.PI * 2;
            const value = dataArray[i];
            const lineLength = dynamicRadius + (value * 0.5);
            
            const x = centerX + Math.cos(angle) * lineLength;
            const y = centerY + Math.sin(angle) * lineLength;
            
            ctx.beginPath();
            ctx.moveTo(centerX + Math.cos(angle) * (dynamicRadius - 10), centerY + Math.sin(angle) * (dynamicRadius - 10));
            ctx.lineTo(x, y);
            ctx.strokeStyle = `rgba(0, 240, 255, ${value / 255})`; // Glow based on volume
            ctx.stroke();
        }
    }
}

// Hook this into your existing "Start" button logic
// In your current LiveKit connection code, after the microphone is enabled:
// const micStream = await room.localParticipant.setMicrophoneEnabled(true);
// setupAudioVisualizer(micStream);

// For testing purposes, if you don't have mic access yet, this creates fake data
// so you can see the visualizer working immediately.
setInterval(() => {
    if (!analyser) {
        // Simulate a voice for testing
        const fakeData = new Uint8Array(128);
        for (let i = 0; i < 128; i++) {
            fakeData[i] = Math.random() * 255;
        }
        // We bypass the analyser for the demo
        if (!dataArray) dataArray = fakeData;
    }
}, 100);
