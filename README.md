# 📌 Advanced Proctoring System  

![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python)  
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-blue?logo=opencv)  
![YOLOv5](https://img.shields.io/badge/YOLOv5-Object_Detection-orange)  

## 🚀 Overview  
The **Advanced Proctoring System** is an AI-powered tool designed to **monitor online exams** by detecting:  
✅ **Phones & electronic devices** using YOLOv5  
✅ **Face absence & multiple faces** to prevent impersonation  
✅ **Eye tracking & looking away detection** using MediaPipe  
✅ **Real-time alerts, logging, and violation tracking**  

## 🛠️ Features  
✔️ **Real-time monitoring** with OpenCV  
✔️ **YOLOv5-based phone detection** (optimized for multiple classes)  
✔️ **Face & eye detection** with Haar cascades  
✔️ **Live audio alerts for violations**  
✔️ **Automated logging & violation reports**  
✔️ **Cross-platform compatibility** (Windows, Linux, macOS)  

## 🔄 System Flowchart  
```mermaid
graph TD;
    A[Start] --> B[Initialize Systems]
    B -->|Load YOLOv5| C[Phone Detection]
    B -->|Load MediaPipe| D[Face & Eye Tracking]
    B -->|Initialize Logging| E[Logging & Reports]

    C -->|Detect Phones| F[Violation Check]
    D -->|Detect Multiple Faces| F
    D -->|Track Eye Movement| F

    F -->|No Violations| G[Continue Monitoring]
    F -->|Violation Detected| H[Trigger Alert & Log]
    H --> I[Capture Screenshot & Record]

    I --> J[Store in proctoring_logs/]
    J --> K[Generate Report]

    G --> B
    H --> B
    K --> L[End]
```

## 📜 Logging & Reports  
📝 **Violation logs** are stored in the `proctoring_logs/` folder.  



## 📌 Future Improvements  
- 🔍 **Enhance accuracy** with deep learning-based face tracking  
- 🎙️ **Voice detection** for verbal cheating detection  
- 📊 **Web-based dashboard** for real-time analytics  

## 🤝 Contributing  
Contributions are welcome! Feel free to open issues or submit pull requests.  

## 📜 License  
This project is licensed under the **MIT License**.  
