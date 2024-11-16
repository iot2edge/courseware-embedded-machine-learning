# Embedded Machine Learning Courseware

[![Markdown link check status badge](https://github.com/edgeimpulse/courseware-embedded-machine-learning/actions/workflows/mlc.yml/badge.svg)](https://github.com/edgeimpulse/courseware-embedded-machine-learning/actions/workflows/mlc.yml) [![Markdown linter status badge](https://github.com/edgeimpulse/courseware-embedded-machine-learning/actions/workflows/mdl.yml/badge.svg)](https://github.com/edgeimpulse/courseware-embedded-machine-learning/actions/workflows/mdl.yml) [![Spellcheck status badge](https://github.com/edgeimpulse/courseware-embedded-machine-learning/actions/workflows/spellcheck.yml/badge.svg)](https://github.com/edgeimpulse/courseware-embedded-machine-learning/actions/workflows/spellcheck.yml) [![HitCount](https://hits.dwyl.com/edgeimpulse/courseware-embedded-machine-learning.svg?style=flat-square&show=unique)](http://hits.dwyl.com/edgeimpulse/courseware-embedded-machine-learning)

Welcome to the Edge Impulse open courseware for embedded machine learning! This repository houses a collection of slides, reading material, project prompts, and sample questions to get you started creating your own embedded machine learning course. You will also have access to videos that cover much of the material. You are welcome to share these videos with your class either in the classroom or let students watch them on their own time.

This repository is part of the Edge Impulse University Program. Please see this page for more information on how to join: [edgeimpulse.com/university](https://edgeimpulse.com/university).

## How to Use This Repository

Please note that the content in this repository is not intended to be a full semester-long course. Rather, you are encouraged to pull from the modules, rearrange the ordering, make modifications, and use as you see fit to integrate the content into your own curriculum.

For example, many of the lectures and examples from the TinyML Courseware (given by [[3]](#3-slides-and-written-material-for-tinyml-courseware-by-harvard-university-is-licensed-under-cc-by-nc-sa-40)) go into detail about how TensorFlow Lite works along with advanced topics like quantization. Feel free to skip those sections if you would just like an overview of embedded machine learning and how to use it with Edge Impulse.

In general, content from [[3]](#3-slides-and-written-material-for-tinyml-courseware-by-harvard-university-is-licensed-under-cc-by-nc-sa-40) cover theory and hands-on Python coding with Jupyter Notebooks to demonstrate these concepts. Content from [[1]](#1-slides-and-written-material-for-introduction-to-embedded-machine-learning-by-edge-impulse-is-licensed-under-cc-by-nc-sa-40) and [[2]](#2-slides-and-written-material-for-computer-vision-with-embedded-machine-learning-by-edge-impulse-is-licensed-under-cc-by-nc-sa-40) cover hands-on demonstrations and projects using Edge Impulse to deploy machine learning models to embedded systems.

Content is divided into separate *modules*. Each module is assumed to be about a week's worth of material, and each section within a module contains about 60 minutes of presentation material. Modules also contain example quiz/test questions, practice problems, and hands-on assignments.

If you would like to see more content than what is available in this repository, please refer to the [Harvard TinyMLedu site](http://tinyml.seas.harvard.edu/) for additional course material.

## License

Unless otherwise noted, slides, sample questions, and project prompts are released under the [Creative Commons Attribution NonCommercial ShareAlike 4.0 International (CC BY-NC-SA 4.0) license](https://creativecommons.org/licenses/by-nc-sa/4.0/). You are welcome to use and modify them for educational purposes.

The YouTube videos in this repository are shared via the standard YouTube license. You are allowed to show them to your class or provide links for students (and others) to watch.

## Professional Development

Much of the material found in this repository is curated from a collection of online courses with permission from the original creators. You are welcome to take the courses (as professional development) to learn the material in a guided fashion or refer students to the courses for additional learning opportunities.

* [Introduction to Embedded Machine Learning](https://www.coursera.org/learn/introduction-to-embedded-machine-learning) - Coursera course by Edge Impulse that introduces neural networks and deep learning concepts and applies them to embedded systems. Hands-on projects rely on training and deploying machine learning models with Edge Impulse. Free with optional paid certificate.
* [Computer Vision with Embedded Machine Learning](https://www.coursera.org/learn/computer-vision-with-embedded-machine-learning) - Follow-on Coursera course that covers image classification and object detection using convolutional neural networks. Hands-on projects rely on training and deploying models with Edge Impulse. Free with optional paid certificate.
* [Tiny Machine Learing (TinyML)](https://www.edx.org/professional-certificate/harvardx-tiny-machine-learning) - EdX course by [Vijay Janapa Reddi](https://scholar.harvard.edu/vijay-janapa-reddi), [Laurence Moroney](https://laurencemoroney.com/), [Pete Warden](https://petewarden.com/), and [Lara Suzuki](https://larissasuzuki.com/). Hands-on projects rely on Python code in Google Colab to train and deploy models to embedded systems with TensorFlow Lite for Microcontrollers. Paid course.

## Prerequisites

Students should be familiar with the following topics to complete the example questions and hands-on assignments:

* **Algebra**
  * Solving linear equations
* **Probability and Statistics**
  * Expressing probabilities of independent events
  * Normal distributions
  * Mean and median
* **Programming**
  * Arduino/C++ programming (conditionals, loops, arrays/buffers, pointers, functions)
  * Python programming (conditionals, loops, arrays, functions, NumPy)

*Optional prerequisites*: many machine learning concepts can be quite advanced. While these advanced topics are briefly discussed in the slides and videos, they are not required for quiz questions and hands-on projects. If you would like to dig deeper into such concepts in your course, students may need to be familiar with the following:

* **Linear algebra**
  * Matrix addition, subtraction, and multiplication
  * Dot product
  * Matrix transposition and inversion
* **Calculus**
  * The derivative and chain rule are important for backpropagation (a part of model training)
  * Integrals and summation are used to find the area under a curve (AUC) for some model evaluations
* **Digital signal processing (DSP)**
  * Sampling rate
  * Nyquist–Shannon sampling theorem
  * Fourier transform and fast Fourier transform (FFT)
  * Spectrogram
* **Machine learning**
  * Logistic regression
  * Neural networks
  * Backpropagation
  * Gradient descent
  * Softmax function
  * K-means clustering
* **Programming**
  * C++ programming (objects, callback functions)
  * Microcontrollers (hardware interrupts, direct memory access, double buffering, real-time operating systems)

## Feedback and Contributing

If you find errors or have suggestions about how to make this material better, please let us know! You may [create an issue](https://github.com/edgeimpulse/course-embedded-machine-learning/issues) describing your feedback or [create a pull request](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request) if you are familiar with Git.

This repo uses automatic link checking and spell checking. If continuous integration (CI) fails after a push, you may find the dead links or misspelled words, fix them, and push again to re-trigger CI. If dead links or misspelled words are false positives (i.e. purposely malformed link or proper noun), you may update [.mlc_config.json](.mlc-config.json) for links to ignore or [.wordlist.txt](.wordlist.txt) for words to ignore.

## Required Hardware and Software

Students will need a computer and Internet access to perform machine learning model training and hands-on exercises with the Edge Impulse Studio and Google Colab. Students are encouraged to use the [Arduino Tiny Machine Learning kit](https://store-usa.arduino.cc/products/arduino-tiny-machine-learning-kit) to practice performing inference on an embedded device.

A Google account is required for [Google Colab](https://colab.research.google.com/).

An Edge Impulse account is required for the [Edge Impulse Studio](https://edgeimpulse.com/).

Students will need to install the latest [Arduino IDE](https://www.arduino.cc/en/software).

## Preexisting Datasets and Projects

This is a collection of preexisting datasets, Edge Impulse projects, or curation tools to help you get started with your own edge machine learning projects. With a public Edge Impulse project, note that you can clone the project to your account and/or download the dataset from the Dashboard.

### Motion

* [Edge Impulse continuous gesture (idle, snake, up-down, wave) dataset](https://docs.edgeimpulse.com/docs/pre-built-datasets/continuous-gestures)
* [Alternate motion (idle, up-down, left-right, circle) project](https://studio.edgeimpulse.com/public/76063/latest)

### Sound

* [Running faucet dataset](https://docs.edgeimpulse.com/docs/pre-built-datasets/running-faucet)
* [Google speech commands dataset](http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz)
* [Keyword spotting dataset curation and augmentation script](https://github.com/ShawnHymel/ei-keyword-spotting/blob/master/ei-audio-dataset-curation.ipynb)
* [Multilingual spoken words corpus](https://mlcommons.org/en/multilingual-spoken-words/)

### Image Classification

* [Electronic components dataset](https://github.com/ShawnHymel/computer-vision-with-embedded-machine-learning/blob/master/Datasets/electronic-components-png.zip?raw=true)
* [Image data augmentation script](https://github.com/ShawnHymel/computer-vision-with-embedded-machine-learning/blob/master/2.3.5%20-%20Project%20-%20Data%20Augmentation/solution_image_data_augmentation.ipynb)

### Object Detection

* [Face detection project](https://studio.edgeimpulse.com/public/87291/latest)

## Syllabus

* [Module 1: Machine Learning on the Edge](#module-1-machine-learning-on-the-edge)
  * [Learning Objectives](#learning-objectives)
  * [Section 1: Machine Learning on the Edge](#section-1-machine-learning-on-the-edge)
  * [Section 2: Limitations and Ethics](#section-2-limitations-and-ethics)
  * [Section 3: Getting Started with Colab](#section-3-getting-started-with-colab)
* [Module 2: Getting Started with Deep Learning](#module-2-getting-started-with-deep-learning)
  * [Learning Objectives](#learning-objectives-1)
  * [Section 1: Machine Learning Paradigm](#section-1-machine-learning-paradigm)
  * [Section 2: Building Blocks of Deep Learning](#section-2-building-blocks-of-deep-learning)
  * [Section 3: Embedded Machine Learning Challenges](#section-3-embedded-machine-learning-challenges)
* [Module 3: Machine Learning Workflow](#module-3-machine-learning-workflow)
  * [Learning Objectives](#learning-objectives-2)
  * [Section 1: Machine Learning Workflow](#section-1-machine-learning-workflow)
  * [Section 2: Data Collection](#section-2-data-collection)
  * [Section 3: Model Training and Evaluation](#section-3-model-training-and-evaluation)
* [Module 4: Model Deployment](#module-4-model-deployment)
  * [Learning Objectives](#learning-objectives-3)
  * [Section 1: Quantization](#section-1-quantization)
  * [Section 2: Embedded Microcontrollers](#section-2-embedded-microcontrollers)
  * [Section 3: Deploying a Model to an Arduino Board](#section-3-deploying-a-model-to-an-arduino-board)
* [Module 5: Anomaly Detection](#module-5-anomaly-detection)
  * [Learning Objectives](#learning-objectives-4)
  * [Section 1: Introduction to Anomaly Detection](#section-1-introduction-to-anomaly-detection)
  * [Section 2: K-means Clustering and Autoencoders](#section-2-k-means-clustering-and-autoencoders)
  * [Section 3: Anomaly Detection in Edge Impulse](#section-3-anomaly-detection-in-edge-impulse)
* [Module 6: Image Classification with Deep Learning](#module-6-image-classification-with-deep-learning)
  * [Learning Objectives](#learning-objectives-5)
  * [Section 1: Image Classification](#section-1-image-classification)
  * [Section 2: Convolutional Neural Network (CNN)](#section-2-convolutional-neural-network-cnn)
  * [Section 3: Analyzing CNNs, Data Augmentation, and Transfer Learning](#section-3-analyzing-cnns-data-augmentation-and-transfer-learning)
* [Module 7: Object Detection and Image Segmentation](#module-7-object-detection-and-image-segmentation)
  * [Learning Objectives](#learning-objectives-6)
  * [Section 1: Introduction to Object Detection](#section-1-introduction-to-object-detection)
  * [Section 2: Image Segmentation and Constrained Object Detection](#section-2-image-segmentation-and-constrained-object-detection)
  * [Section 3: Responsible AI](#section-3-responsible-ai)
* [Module 8: Keyword Spotting](#module-8-keyword-spotting)
  * [Learning Objectives](#learning-objectives-7)
  * [Section 1: Audio Classification](#section-1-audio-classification)
  * [Section 2: Spectrograms and MFCCs](#section-2-spectrograms-and-mfccs)
  * [Section 3: Deploying a Keyword Spotting System](#section-3-deploying-a-keyword-spotting-system)

## Course Material

### Module 1: Machine Learning on the Edge

This module provides an overview of machine learning and how it can be used to solve problems. It also introduces the idea of running machine learning algorithms on resource-constrained devices, such as microcontrollers. It covers some of the limitations and ethical concerns of machine learning. Finally, it demonstrates a few examples of Python in Google Colab, which will be used in early modules to showcase how programming is often performed for machine learning with TensorFlow and Keras.

#### Learning Objectives

1. Describe the differences between artificial intelligence, machine learning, and deep learning
2. Provide examples of how machine learning can be used to solve problems (that traditional deterministic programming cannot)
3. Provide examples of how embedded machine learning can be used to solve problems (where other forms of machine learning would be limited or inappropriate)
4. Describe the limitations of machine learning
5. Describe the ethical concerns of machine learning
6. Describe the differences between supervised and unsupervised machine learning

#### Section 1: Machine Learning on the Edge

##### Lecture Material

| ID | Description | Links | Attribution |
|----|-------------|:-----:|:-----------:|
| 1.1.1 | What is machine learning | [![What is machine learning](https://www.youtube.com/watch?v=yjprpOoH5c81)](https://www.youtube.com/watch?v=RDGCGho5oaQ&list=PL7VEa1KauMQqZFj_nWRfsCZNXaBbkuurG&index=3) [slides](Module%201%20-%20Introduction%20to%20Machine%20Learning/1.1.1.what-is-machine-learning.1.pptx?raw=true) | [[1]](#1-slides-and-written-material-for-introduction-to-embedded-machine-learning-by-edge-impulse-is-licensed-under-cc-by-nc-sa-40) |
| 1.1.2 | Machine learning on embedded devices | [![Machine learning on embedded devices](https://www.youtube.com/watch?v=yjprpOoH5c82)](https://www.youtube.com/watch?v=Thg_EK9xxVk&list=PL7VEa1KauMQqZFj_nWRfsCZNXaBbkuurG&index=6) [slides](Module%201%20-%20Introduction%20to%20Machine%20Learning/1.1.2.machine-learning-on-embedded-devices.1.pptx?raw=true) | [[1]](#1-slides-and-written-material-for-introduction-to-embedded-machine-learning-by-edge-impulse-is-licensed-under-cc-by-nc-sa-40) |
| 1.1.3 | What is tiny machine learning | [slides](Module%201%20-%20Introduction%20to%20Machine%20Learning/1.1.3.what-is-tiny-machine-learning.3.pptx?raw=true) | [[3]](#3-slides-and-written
