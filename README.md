# 🌟 Local **Super-Agents**: ***Secure** Practical AGI for the **Real World***

![](https://github.com/Independent-AI-Labs/local-super-agents/blob/main/res/built_on.png)

Privacy-first agentic framework with powerful reasoning & task automation features. Natively distributed and fully **ISO 27XXX** compliant.

---

## 🏗️ Development Approach

We believe that the future of software is defined by **security-driven development**, **evolving architectures**, and **on-demand functionality** that can be rapidly deployed, scaled, and replaced as needed. This, and a shift towards **distributed** workloads emphasize granular access controls, continuous monitoring, and **Zero Trust** principles to ensure system integrity and regulatory compliance.


---

## **🔑 Key Features**

---

### **STATUS**

| ✅ **STABLE RELEASE** | ☑️ **DEV RELEASE** | 🧪 **QA / TESTING** | 🚧 **IN PROGRESS** | 📋 **PLANNED** | 💬 **DISCUSSION** |

---
### **✨ Latest Open-Source Language, Speech & Vision Models**
   - 🧪 Seamless integration with **Open WebUI**, **Ollama**, and **vLLM**
   - 🚧 Real-time voice conversations with **Whisper & Kokoro**
   - 💬 Image / video generation features and automation

---

### **🧠 100% Transparent Machine Reasoning & Chain-of-Though Generation**
   - 🚧 Use your AI assistant to create **interactive tasks**, monitor agent activity and manage platform operations
   - 🚧 **Pause**, **resume**, **replay**, **clone** and execute tasks locally and within secure distributed environments
   - 🚧 Modular **Alignment & Reasoning** core system, allowing granular control over **guardrails**, **CoT** models, templates, and more
   - 🚧 Emulates human cognitive processes for natural task / activity comprehension, **intelligent context building** and user interpretability
   - 🚧 Facilitates complex **issue decomposition**, **goal identification** and **interactive solution generation** for real-world problem-solving
   - 🚧 Architectural paper coming soon!

![](https://github.com/Independent-AI-Labs/local-super-agents/blob/main/res/task_decomposition.png)
---

### **🚅 High-Performance Text Analysis, User Memory & Model Tuning Features**
   - 🧪 CPU-optimized text document indexing & semantic search through **High-Yield Pattern Extraction** (**"HYPE"** - document and benchmarks coming soon)  
   - 🚧 **Continuous analysis** of operational metadata, conversation transcripts and user instructions for **context-optimal memory retrieval** and dynamic alignment with user preference
   - 🚧 Automated dataset curation for **test-time training**, model **re-training** and **fine-tuning** (including support for **LoRA**, **QLoRA**, **DPO**, **QA-LoRA** and **ReLoRA**)

---

### **🌌 Secure Distributed System**
   - 🚧 **Node auto-discovery**, **load-balancing** and **federated work** with **PySyft**
   - 🚧 **Multi-host** model weight distribution for inference and fine-tuning via **DeepSeed** 
   - 📋 **Distributed** training for different model architectures

---

### **🔒 Comprehensive Security & Compliance**
   - 🧪 **Vaultwarden** integration for agent credential sharing (support for other secret managers coming soon)
   - 🚧 **Sandboxed agent file manipulation** and supervised synchronization with host filesystem
   - 🚧 Designed to conform with **ISO/IEC 27001, 27017, 27018** and **NIST CSF**
   - 📋 Built-in process, data, and agent **monitoring** with **real-time intervention** capabilities
   - 📋 Secure network-wide document **replication** & step-by-step file **rollback**
   - 💬 **Sandboxed code generation, testing & execution** with automated static analysis via **Betterscan**

---

### **⚡ Extensive AI Accelerator Support**
   - 🧪 **LLM** / **VLM** inference and multi-GPU model distribution for **Intel Arc** and other **oneAPI** / **SYCL** devices
   - 🚧 Support for **NVIDIA** **RTX** Cards / **CUDA**
   - 📋 Support for **AMD** **Radeon** GPUs & NPUs (TBD)

---

### **💻 Secure Remote & Internet Access**
   - 🧪 Self-hosted web access with optional automated **OpenVPN** and **Cloudflare** tunneling functionality
   - 🧪 System-wide **tracking** and **advertisement** protection, reverse proxy support for web access
   - 🧪 Multi-process search engine aggregation & agent web surfer built on **SearXNG**, **Chrome Engine** and **Selenium**
   - 🧪 Automated web browsing that **avoids bot detection**, handles **client-side rendered data** and digests relevant content
   - 📋 Low-latency web-based **4K desktop streaming** for remote work, collaboration and agent supervision
    
---

### **🪟 Complete Windows Integration**
   - 🚧 **Streamlined** installation on **Windows** systems & official auto-update channels
   - 🚧 Floating, fullscreen & desktop AI assistant app modes (**Open WebUI**)
   - 🚧 **System / network activity feed** with notifications for agent actions and file syncs
   - 🚧 **User-controlled agent access** to any application window (view / read only)
   - 💬 Platform launcher / dashboard GUI  
   - 💬 **General computer and browser use** for agents, **supervised** desktop and application management 

![](https://github.com/Independent-AI-Labs/local-super-agents/blob/main/res/screens/floating_assist.png)

---

### **🐧 KDE / Gnome Integrations**
   - ❤️ Coming Soon

---

### **📈 Productivity Features**
   - 🚧 Read, compose and send emails via **SMTP** 
   - 🚧 **Google Docs** and **Microsoft 365** integrations for document-based collaboration
   - 🚧 Support for agent language & localization features via dedicated module & translation models
   - 📋 **Connect to any service** - automated universal **REST API** client generation and integration (initial support for **OpenAPI** / **Swagger**)
   - 📋 Quantitative analysis and  visualization of data with **Python**, **Bokeh** and **Plotly**
   - 💬 **NextCloud** integration


---

## **👥 Crowd & Distributed Training Aspirations**

### *🚨 Help us build the first SotA language model trained on crowd-sourced hardware!*
We are looking for active contributors that can assist with:

- Automated large-scale model training with minimal overhead
- Real-time monitoring of host resource usage across the network (bandwidth, memory, CPU, GPU, etc.)
- Dynamic resource allocation and redundancy
- Flexible checkpointing for mid-training evaluations
- Federated learning workflows to ensure privacy and efficiency
- **ARM** device and OS support (**Mac**, **Android 16+**, **Linux** & **Windows**)

---

## **🚀 HYPE Performance Benchmarks** (Coming Soon)

In a few days, we will be kicking-off the public repo with code and runnable benchmarks for our built-in **semantic search** solution powering many of the platform's features.
Some fuzzy matching performance numbers are presented below. The tests were conducted on **Intel Core i9-14900F** with **24 physical** CPU cores and a **Kingston NV2 M.2 SSD**.

### **Structured Data Search (20 unique terms + scoring)**

| Test                                                | Time (ms)   | Throughput                                                                                         |
|-----------------------------------------------------|-------------|----------------------------------------------------------------------------------------------------|
| **Single-Core Item Search**                         | 6614.97     | **Raw Data:** 154.8 MB/s <br/>**Items:** 844,059.46 items/s                                         |
| **Multi-Core Item Search**<br/>**Scaling:** +32% / Core | 10551.87    | **Raw Data:** 1,223.0 MB/s <br/>**Items:** 5,224,696.88 items/s          |

### **Document Search (20 unique terms + scoring)**

| Test                                                         | Time (s)    | Throughput                                                                                          |
|--------------------------------------------------------------|------------|-----------------------------------------------------------------------------------------------------|
| **Single-Core Document Search**                              | 6530.47    | **Raw Data:** 144.59 MB/s <br/>**Documents:** 2,556.94 docs/s                                       |
| **Multi-Core Document Search**<br/>**Scaling:** +39%  / Core | 16722.97   | **Raw Data:** 1355.17 MB/s <br/>**Documents:** 23,964.16 docs/s           |
|                                                              |             |                                                                                                     |

> **Benchmark Datasets (pre-indexed, will be included):**  
> - *Mother of All Movie Reviews* (structured, ~56M unique rows, ~1.3B searched)  
> - *EUR-Lex Energy Legislation* (average ~500 KB each, ~17K unique documents, ~400K searched)

> **CPU Core Scaling:**  
> - Average core scaling will depend on many factors including CPU architecture, power budget, available cache, etc. Scaling beyond physical cores is usually very detrimental to performance.
> - E.g. on heterogeneous, power limited platforms like the i9-14900F, the per-core scaling factor rapidly drops from ~80% at 2 concurrent processes to <60% at 8 and below 25% when all logical cores are used...
> - Disk read times also have a significant impact, especially with structured data. Naturally, you'll need a high-performance SSD to get the most out of the solution.
> - Generally, server-class CPUs are more consistent with their scaling and consumer-grade chips tend to have better single-core performance.
---

## **📅 Roadmap**

> TBD

---

## **🤝 Stay Connected**

> - **Regular Updates**  
  We aim to roll out new features, optimizations, and integrations to keep the project at the forefront of AI development.

> - **Collaborative Environment**  
  We welcome contributions from the community, whether through bug reports, feature requests, or direct involvement in the codebase.
