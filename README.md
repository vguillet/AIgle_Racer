# AIgle Racer Project

![](https://github.com/vguillet/AIgle_Racer/blob/master/AIgle_Hardware/Renders/DoorHighres.jpg)

> This project was undertaken with the intention of paving the way to future drone racing endeavor, and facilitate AI research and development.

> The design team consists of eleven Bachelor students working full time for 10 weeks. 

Drone racing has gathered increasing attention and interest in the past few years, establishing itself as a sport on par with more traditional ones such as Formula 1. A subset of it, autonomous drone racing, has been growing rapidly in popularity, recently capturing the interest of both the industry, the general public, and the scientific community. Since its beginning in 2016 with the IROS Autonomous Drone Race, the number of competitions have expanded, with more industry players such as the Drone Racing League (DRL) hosting its own competition in 2019, to fully virtual drone races being held by NeurIPS (also in 2019). 

The essence of these competition has however remained the same. Teams compete to develop the most effective software to pilot a standardised drone as quickly as possible over a given circuit. The drone being standardised is a critical aspect of the competition, as it ensures that the software is the sole deciding factor. During the DRL races, the competition host provided the drones competitors would be flying, ensuring this way that no drone had a mechanical advantage over its opponents. This however brings about a number of issues. The cost of hosting a competition goes up drastically, as the cost of producing, maintaining, and repairing the drones then becomes the responsibility of the host. It also means competitors would not be able to effectively test, adjust, and optimise their algorithms prior to the race, reducing the teams' potential ability to compete. In an effort to alleviate these issues, group 19 of the TU Delft 2020 DSE aims to propose an open-source hardware solution for race organizers and teams to be used in future autonomous drone racing competitions. The end goal of the project is to facilitate a worldwide collaboration on the development of artificial intelligence in agile autonomous drone flight. 

This repository contains all the information and results necessary to get started with the AIgle Racer platform. 

## AIgle Documentation

The AIgle Racer is documented in four distinct reports, each compiled at various stages of progress of the project.
  - Project plan: Depicts the initial organisation of the project
  - Baseline report: Describes the structuration of the research and development based on preliminary research,
  - Mid-term report: Explains the concepts elaboration and evaluation
  - Final report: Contains the detailed design, with the fully developed concept and instructions based on in-depth research.

![](https://github.com/vguillet/AIgle_Racer/blob/master/Misc/Documentation_pane.png)

## AIgle Hardware
The AIgle hardware folder contains all the information necessary for the production and assembly of the AIgle drone. To guide and facilitate the production, the following are provided:

  - Parts lists
  - 3D Catia models
  - Custom PCB diagrams and description

## AIgle Software
> Note: The AIgle demonstration software can be found in AIgle_Project/src

The AIgle Project src file contains all the demonstrations software provided to act as a baseline for building autonomous racing software to use with the AIgle racer platform. A general suggested pipeline is available in the documentation along with the result of research performed on the various methods and approach demonstrated.
  - Machine Vision & State estimation: Focus on retrieving and processing sensor data to get an estimate of the position of the drone in 3D space
  - Navigation: Focus on determining the path and trajectories to be followed by the drone during the race. Algorithms available: DDQN, DDPG
  - Control & stability: Focus on translating the desired drone movement (control position, velocities, attitude) into commands (subsequenctly sent to an electronic speed controller) 

## AIgle Concept Development Tools
The concept evelopment tool folder contains the code which was used for desiging and sizing the drone. This includes:
  - Obtain suitable battery/motor/propelor triplets based on different input criterias
  - Vertification code, to be used with [eCalc](https://ecalc.ch/xcoptercalc.php)

## Acknowledgement
The group would like to extend special thanks to Ir. C. De Wagter for his tremendous insight into drone racing, and for sharing his knowledge and experience in drone racing design. The group would also like to thank B. Mercier and Y. Zhang for their valuable advices and help during the design phase, and D. Martini for his important input throughout the project. The tutors and coaches have been present throughout the entire development of the project, and their contribution has proven essential to its success. 


