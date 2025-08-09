import { setRobotCenter, setTargetGripperPosition } from './robot.js';
import { animate } from './animation.js';
import { setupEventListeners } from './event-listeners.js';
import { ARM_LENGTH } from './config.js';

const canvas = document.getElementById('robotCanvas');
const ctx = canvas.getContext('2d');
let mainContainer, robotPanel;

function setupCanvasSize() {
    if (!robotPanel) {
        robotPanel = document.querySelector('.robot-panel');
        mainContainer = document.querySelector('.main-container');
    }
    canvas.width = robotPanel.offsetWidth;
    canvas.height = robotPanel.offsetHeight;
    setRobotCenter(canvas.width * 4 / 5, canvas.height * 1 / 4);
}

async function getRobotCommand() {
    let res = await fetch("/classify");
    let data = await res.json();
    if (data.error) {
            console.error("Server error:", data.error);
            alert("Error: " + data.error); // Thông báo cho người dùng
            return;
        }
    console.log("Detected:", data.trash_type, "Mode:", data.mode);

    console.log("Detected:", data.trash_type, "Mode:", data.mode);

    if (data.mode === "pickup_and_drop") {
        moveToCenter();
        setTimeout(() => {
            grabTrash();
            setTimeout(() => {
                moveToBin(data.coords.drop);
                setTimeout(() => {
                    releaseTrash();
                }, 1500);
            }, 1000);
        }, 1500);
    }
}

document.addEventListener('DOMContentLoaded', () => {
    setupCanvasSize();
    setupEventListeners(canvas, mainContainer, robotPanel);
    animate(ctx, canvas); // vẽ robot
    setInterval(getRobotCommand, 5000); // lấy lệnh từ Python
});

// ===== Hàm điều khiển robot =====
function moveToCenter() {
    console.log("Robot: Di chuyển đến giữa camera");
    setTargetGripperPosition(- canvas.width * 3 / 7, canvas.height * 1 / 5);
}

function grabTrash() {
    console.log("Robot: Gắp rác");
}

function moveToBin(binCoords) {
    console.log("Robot: Di chuyển tới thùng rác");
    setTargetGripperPosition(binCoords[0] * canvas.width, binCoords[1] * canvas.height);
}

function releaseTrash() {
    console.log("Robot: Thả rác");
}
