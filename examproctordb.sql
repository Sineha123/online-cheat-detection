-- phpMyAdmin SQL Dump
-- version 5.2.1
-- https://www.phpmyadmin.net/
--
-- Host: 127.0.0.1
-- Generation Time: Mar 04, 2026 at 08:07 PM
-- Server version: 10.4.32-MariaDB
-- PHP Version: 8.0.30

SET SQL_MODE = "NO_AUTO_VALUE_ON_ZERO";
START TRANSACTION;
SET time_zone = "+00:00";


/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!40101 SET NAMES utf8mb4 */;

--
-- Database: `examproctordb`
--

-- --------------------------------------------------------

--
-- Table structure for table `exam_results`
--

CREATE TABLE `exam_results` (
  `ResultID` int(11) NOT NULL,
  `StudentID` int(11) NOT NULL,
  `SessionID` int(11) NOT NULL,
  `Score` decimal(5,2) DEFAULT 0.00,
  `TotalQuestions` int(11) DEFAULT 10,
  `CorrectAnswers` int(11) DEFAULT 0,
  `SubmissionTime` datetime DEFAULT current_timestamp(),
  `Status` enum('PASS','FAIL','TERMINATED') DEFAULT 'FAIL'
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

--
-- Dumping data for table `exam_results`
--

INSERT INTO `exam_results` (`ResultID`, `StudentID`, `SessionID`, `Score`, `TotalQuestions`, `CorrectAnswers`, `SubmissionTime`, `Status`) VALUES
(5, 12, 5, 0.00, 10, 0, '2026-03-04 04:35:20', 'TERMINATED'),
(6, 12, 6, 190.00, 10, 19, '2026-03-04 04:47:16', 'PASS'),
(14, 13, 15, 0.00, 25, 0, '2026-03-04 06:26:11', 'TERMINATED'),
(29, 14, 30, 1.60, 125, 2, '2026-03-04 23:23:36', 'TERMINATED');

-- --------------------------------------------------------

--
-- Table structure for table `exam_sessions`
--

CREATE TABLE `exam_sessions` (
  `SessionID` int(11) NOT NULL,
  `StudentID` int(11) NOT NULL,
  `StartTime` datetime DEFAULT current_timestamp(),
  `EndTime` datetime DEFAULT NULL,
  `Status` enum('IN_PROGRESS','COMPLETED','TERMINATED') DEFAULT 'IN_PROGRESS'
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

--
-- Dumping data for table `exam_sessions`
--

INSERT INTO `exam_sessions` (`SessionID`, `StudentID`, `StartTime`, `EndTime`, `Status`) VALUES
(5, 12, '2026-03-04 04:35:02', '2026-03-04 04:35:20', 'TERMINATED'),
(6, 12, '2026-03-04 04:42:13', '2026-03-04 04:47:16', 'COMPLETED'),
(7, 13, '2026-03-04 05:04:21', '2026-03-04 05:04:34', 'TERMINATED'),
(8, 13, '2026-03-04 05:05:24', '2026-03-04 05:05:39', 'TERMINATED'),
(9, 13, '2026-03-04 05:07:31', '2026-03-04 05:09:09', 'TERMINATED'),
(10, 13, '2026-03-04 05:55:26', '2026-03-04 05:58:30', 'COMPLETED'),
(11, 13, '2026-03-04 06:12:08', '2026-03-04 06:13:27', 'TERMINATED'),
(12, 13, '2026-03-04 06:17:27', '2026-03-04 06:18:25', 'TERMINATED'),
(13, 13, '2026-03-04 06:24:46', '2026-03-04 06:25:01', 'TERMINATED'),
(14, 13, '2026-03-04 06:25:50', '2026-03-04 06:25:54', 'COMPLETED'),
(15, 13, '2026-03-04 06:25:54', '2026-03-04 06:26:11', 'TERMINATED'),
(16, 14, '2026-03-04 20:25:03', '2026-03-04 20:25:16', 'TERMINATED'),
(17, 14, '2026-03-04 20:28:18', '2026-03-04 20:28:40', 'TERMINATED'),
(18, 14, '2026-03-04 20:39:08', '2026-03-04 20:39:21', 'TERMINATED'),
(19, 14, '2026-03-04 20:41:09', '2026-03-04 20:41:23', 'TERMINATED'),
(20, 14, '2026-03-04 20:41:47', '2026-03-04 20:42:09', 'TERMINATED'),
(21, 14, '2026-03-04 20:56:13', '2026-03-04 20:56:44', 'TERMINATED'),
(22, 14, '2026-03-04 20:57:19', '2026-03-04 20:58:00', 'TERMINATED'),
(23, 14, '2026-03-04 22:09:14', '2026-03-04 22:09:57', 'TERMINATED'),
(24, 14, '2026-03-04 22:21:17', '2026-03-04 22:21:40', 'TERMINATED'),
(25, 14, '2026-03-04 22:22:47', '2026-03-04 22:23:18', 'TERMINATED'),
(26, 14, '2026-03-04 22:36:12', '2026-03-04 22:36:38', 'TERMINATED'),
(27, 14, '2026-03-04 22:43:48', '2026-03-04 22:44:32', 'TERMINATED'),
(28, 14, '2026-03-04 22:53:52', '2026-03-04 22:54:36', 'TERMINATED'),
(29, 14, '2026-03-04 22:55:05', '2026-03-04 22:55:57', 'TERMINATED'),
(30, 14, '2026-03-04 23:22:15', '2026-03-04 23:23:36', 'TERMINATED');

-- --------------------------------------------------------

--
-- Table structure for table `profiles`
--

CREATE TABLE `profiles` (
  `id` int(11) NOT NULL,
  `student_id` int(11) NOT NULL,
  `profile_image_path` varchar(255) NOT NULL,
  `image_type` enum('upload','webcam') DEFAULT 'upload',
  `face_detected` tinyint(1) DEFAULT 0,
  `created_at` datetime DEFAULT current_timestamp(),
  `updated_at` datetime DEFAULT current_timestamp() ON UPDATE current_timestamp()
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

-- --------------------------------------------------------

--
-- Table structure for table `students`
--

CREATE TABLE `students` (
  `ID` int(11) NOT NULL,
  `Name` varchar(100) NOT NULL,
  `username` varchar(100) DEFAULT NULL,
  `Email` varchar(100) NOT NULL,
  `Password` varchar(255) NOT NULL,
  `Profile` varchar(255) DEFAULT NULL,
  `Role` enum('ADMIN','STUDENT') NOT NULL DEFAULT 'STUDENT'
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

--
-- Dumping data for table `students`
--

INSERT INTO `students` (`ID`, `Name`, `username`, `Email`, `Password`, `Profile`, `Role`) VALUES
(1, 'Admin User', NULL, 'admin@gmail.com', 'pbkdf2:sha256:600000$kxif5EZR92sBNNLI$027178a9641a1a5246360d13ed9f43b7749626c723f473aa03d59df87209d837', NULL, 'ADMIN'),
(2, 'Test Student', NULL, 'student@test.com', '12345', 'face_2_1772578100.jpg', 'STUDENT'),
(3, 'student1', NULL, 'student@exam.com', '123456', NULL, 'STUDENT'),
(4, 'username', NULL, 'student1@exam.com', '123456', NULL, 'STUDENT'),
(8, 'test123', NULL, 'test123@gmail.com', '123456', NULL, 'STUDENT'),
(9, 'tester', NULL, 'tester1@gmail.com', '123456', 'tester1@gmail.com_image.jpeg', 'STUDENT'),
(10, 'leela', NULL, 'leela@gmail.com', '123456', 'leela@gmail.com_WhatsApp_Image_2025-11-13_at_00.58.45_23b207fa.jpg', 'STUDENT'),
(12, 'kashish', NULL, 'kashish@gmail.com', 'pbkdf2:sha256:600000$eeA8UFiySx1fOfbj$858e3c9d5475f5d07bc26437821edb5a7d18250a7d5d77b0d32945accd9da482', 'kashish@gmail.com_WhatsApp_Image_2026-01-08_at_1.01.20_AM.jpeg', 'STUDENT'),
(13, 'newtest', NULL, 'newtester@gmail.com', 'pbkdf2:sha256:600000$sH4EhURaaI5bG2m8$8bc20ceda32cdab171de7500633f458f9c1018be60aff00f61548fcc2511e881', 'newtester@gmail.com_archetypal-female-_3249633c.webp', 'STUDENT'),
(14, 'Suhani', NULL, 'suhani@gmail.com', 'pbkdf2:sha256:600000$0hTIE9wtr3vLKQM4$1775cd3e697742ad0ff2430e50b2f27a933093748efaa26d22fec26028218514', 'suhani@gmail.com_webcam_20260304_202354.jpg', 'STUDENT');

-- --------------------------------------------------------

--
-- Table structure for table `violations`
--

CREATE TABLE `violations` (
  `ViolationID` int(11) NOT NULL,
  `StudentID` int(11) NOT NULL,
  `SessionID` int(11) NOT NULL,
  `ViolationType` varchar(64) NOT NULL,
  `Details` text DEFAULT NULL,
  `Timestamp` datetime DEFAULT current_timestamp()
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

--
-- Dumping data for table `violations`
--

INSERT INTO `violations` (`ViolationID`, `StudentID`, `SessionID`, `ViolationType`, `Details`, `Timestamp`) VALUES
(7, 12, 5, 'MULTIPLE_FACES', '???? Multiple Faces detected by camera', '2026-03-04 04:35:20'),
(8, 12, 5, 'MULTIPLE_FACES', '???? Multiple Faces detected by camera', '2026-03-04 04:35:20'),
(9, 12, 5, 'MULTIPLE_FACES', '???? Multiple Faces detected by camera', '2026-03-04 04:35:20'),
(10, 12, 6, 'TAB_SWITCH', 'Switched away from exam tab (1 times)', '2026-03-04 04:43:12'),
(11, 12, 6, 'TAB_SWITCH', 'Switched away from exam tab (1 times)', '2026-03-04 04:47:16'),
(12, 13, 7, 'TAB_SWITCH', 'Switched away from exam tab (1 times)', '2026-03-04 05:04:23'),
(13, 13, 7, 'VOICE_DETECTED', 'Voice/noise detected from microphone', '2026-03-04 05:04:25'),
(14, 13, 7, 'VOICE_DETECTED', 'Voice/noise detected from microphone', '2026-03-04 05:04:30'),
(15, 13, 7, 'TAB_SWITCH', 'Switched away from exam tab (1 times)', '2026-03-04 05:04:34'),
(16, 13, 7, 'VOICE_DETECTED', 'Voice/noise detected from microphone', '2026-03-04 05:04:34'),
(17, 13, 7, 'VOICE_DETECTED', 'Voice/noise detected from microphone', '2026-03-04 05:04:34'),
(18, 13, 8, 'VOICE_DETECTED', 'Voice/noise detected from microphone', '2026-03-04 05:05:26'),
(19, 13, 8, 'VOICE_DETECTED', 'Voice/noise detected from microphone', '2026-03-04 05:05:31'),
(20, 13, 8, 'VOICE_DETECTED', 'Voice/noise detected from microphone', '2026-03-04 05:05:36'),
(21, 13, 8, 'VOICE_DETECTED', 'Voice/noise detected from microphone', '2026-03-04 05:05:39'),
(22, 13, 8, 'VOICE_DETECTED', 'Voice/noise detected from microphone', '2026-03-04 05:05:39'),
(23, 13, 8, 'VOICE_DETECTED', 'Voice/noise detected from microphone', '2026-03-04 05:05:39'),
(24, 13, 9, 'VOICE_DETECTED', 'Human voice detected from microphone', '2026-03-04 05:08:23'),
(25, 13, 9, 'VOICE_DETECTED', 'Human voice detected from microphone', '2026-03-04 05:08:31'),
(26, 13, 9, 'TAB_SWITCH', 'Switched away from exam tab (1 times)', '2026-03-04 05:09:06'),
(27, 13, 9, 'VOICE_DETECTED', 'Human voice detected from microphone', '2026-03-04 05:09:09'),
(28, 13, 9, 'VOICE_DETECTED', 'Human voice detected from microphone', '2026-03-04 05:09:09'),
(29, 13, 9, 'TAB_SWITCH', 'Switched away from exam tab (1 times)', '2026-03-04 05:09:09'),
(30, 13, 10, 'VOICE_DETECTED', 'Microphone muted/unavailable', '2026-03-04 05:56:01'),
(31, 13, 10, 'VOICE_DETECTED', 'Microphone muted/unavailable', '2026-03-04 05:58:30'),
(32, 13, 11, 'VOICE_DETECTED', 'Microphone muted/unavailable', '2026-03-04 06:12:15'),
(33, 13, 11, 'TAB_SWITCH', 'Switched away from exam tab (1 times)', '2026-03-04 06:12:35'),
(34, 13, 11, 'TAB_SWITCH', 'Switched away from exam tab (2 times)', '2026-03-04 06:13:24'),
(35, 13, 11, 'VOICE_DETECTED', 'Microphone muted/unavailable', '2026-03-04 06:13:27'),
(36, 13, 11, 'TAB_SWITCH', 'Switched away from exam tab (1 times)', '2026-03-04 06:13:27'),
(37, 13, 11, 'TAB_SWITCH', 'Switched away from exam tab (2 times)', '2026-03-04 06:13:27'),
(38, 13, 12, 'VOICE_DETECTED', 'Microphone muted/unavailable', '2026-03-04 06:17:35'),
(39, 13, 12, 'TAB_SWITCH', 'Switched away from exam tab (1 times)', '2026-03-04 06:18:16'),
(40, 13, 12, 'TAB_SWITCH', 'Switched away from exam tab (2 times)', '2026-03-04 06:18:22'),
(41, 13, 12, 'VOICE_DETECTED', 'Microphone muted/unavailable', '2026-03-04 06:18:25'),
(42, 13, 12, 'TAB_SWITCH', 'Switched away from exam tab (1 times)', '2026-03-04 06:18:25'),
(43, 13, 12, 'TAB_SWITCH', 'Switched away from exam tab (2 times)', '2026-03-04 06:18:25'),
(44, 13, 13, 'NO_FACE', '???? No Face Detected detected by camera', '2026-03-04 06:25:01'),
(45, 13, 13, 'VOICE_DETECTED', '???? Voice/Noise Detected detected by camera', '2026-03-04 06:25:01'),
(46, 13, 13, 'NO_FACE', '???? No Face Detected detected by camera', '2026-03-04 06:25:01'),
(47, 13, 15, 'VOICE_DETECTED', '???? Voice/Noise Detected detected by camera', '2026-03-04 06:26:11'),
(48, 13, 15, 'NO_FACE', '???? No Face Detected detected by camera', '2026-03-04 06:26:11'),
(49, 13, 15, 'NO_FACE', '???? No Face Detected detected by camera', '2026-03-04 06:26:11'),
(50, 14, 16, 'NO_FACE', '???? No Face Detected detected by camera', '2026-03-04 20:25:16'),
(51, 14, 16, 'NO_FACE', '???? No Face Detected detected by camera', '2026-03-04 20:25:16'),
(52, 14, 16, 'NO_FACE', '???? No Face Detected detected by camera', '2026-03-04 20:25:16'),
(53, 14, 16, 'VOICE_DETECTED', '???? Voice/Noise Detected detected by camera', '2026-03-04 20:25:16'),
(54, 14, 17, 'NO_FACE', '???? No Face Detected detected by camera', '2026-03-04 20:28:40'),
(55, 14, 17, 'NO_FACE', '???? No Face Detected detected by camera', '2026-03-04 20:28:40'),
(56, 14, 17, 'NO_FACE', '???? No Face Detected detected by camera', '2026-03-04 20:28:40'),
(57, 14, 18, 'NO_FACE', '???? No Face Detected detected by camera', '2026-03-04 20:39:21'),
(58, 14, 18, 'NO_FACE', '???? No Face Detected detected by camera', '2026-03-04 20:39:21'),
(59, 14, 18, 'NO_FACE', '???? No Face Detected detected by camera', '2026-03-04 20:39:21'),
(60, 14, 19, 'VOICE_DETECTED', '???? Voice/Noise Detected detected by camera', '2026-03-04 20:41:23'),
(61, 14, 19, 'VOICE_DETECTED', '???? Voice/Noise Detected detected by camera', '2026-03-04 20:41:23'),
(62, 14, 19, 'NO_FACE', '???? No Face Detected detected by camera', '2026-03-04 20:41:23'),
(63, 14, 20, 'NO_FACE', '???? No Face Detected detected by camera', '2026-03-04 20:42:09'),
(64, 14, 20, 'NO_FACE', '???? No Face Detected detected by camera', '2026-03-04 20:42:09'),
(65, 14, 20, 'NO_FACE', '???? No Face Detected detected by camera', '2026-03-04 20:42:09'),
(66, 14, 21, 'VOICE_DETECTED', '???? Voice/Noise Detected detected by camera', '2026-03-04 20:56:44'),
(67, 14, 21, 'VOICE_DETECTED', '???? Voice/Noise Detected detected by camera', '2026-03-04 20:56:44'),
(68, 14, 21, 'NO_FACE', '???? No Face Detected detected by camera', '2026-03-04 20:56:44'),
(69, 14, 22, 'VOICE_DETECTED', 'Continuous human voice detected from microphone', '2026-03-04 20:57:30'),
(70, 14, 22, 'VOICE_DETECTED', 'Continuous human voice detected from microphone', '2026-03-04 20:58:00'),
(71, 14, 22, 'VOICE_DETECTED', '???? Voice/Noise Detected detected by camera', '2026-03-04 20:58:00'),
(72, 14, 22, 'VOICE_DETECTED', '???? Voice/Noise Detected detected by camera', '2026-03-04 20:58:00'),
(73, 14, 23, 'VOICE_DETECTED', '???? Voice/Noise Detected detected by camera', '2026-03-04 22:09:57'),
(74, 14, 23, 'NO_FACE', '???? No Face Detected detected by camera', '2026-03-04 22:09:57'),
(75, 14, 23, 'VOICE_DETECTED', '???? Voice/Noise Detected detected by camera', '2026-03-04 22:09:57'),
(76, 14, 24, 'VOICE_DETECTED', '???? Voice/Noise Detected detected by camera', '2026-03-04 22:21:40'),
(77, 14, 24, 'VOICE_DETECTED', '???? Voice/Noise Detected detected by camera', '2026-03-04 22:21:40'),
(78, 14, 24, 'VOICE_DETECTED', '???? Voice/Noise Detected detected by camera', '2026-03-04 22:21:40'),
(79, 14, 25, 'VOICE_DETECTED', 'Continuous human voice detected from microphone', '2026-03-04 22:22:56'),
(80, 14, 25, 'VOICE_DETECTED', 'Continuous human voice detected from microphone', '2026-03-04 22:23:18'),
(81, 14, 25, 'VOICE_DETECTED', '???? Voice/Noise Detected detected by camera', '2026-03-04 22:23:18'),
(82, 14, 25, 'VOICE_DETECTED', '???? Voice/Noise Detected detected by camera', '2026-03-04 22:23:18'),
(83, 14, 26, 'VOICE_DETECTED', '???? Voice/Noise Detected detected by camera', '2026-03-04 22:36:38'),
(84, 14, 26, 'VOICE_DETECTED', '???? Voice/Noise Detected detected by camera', '2026-03-04 22:36:38'),
(85, 14, 26, 'VOICE_DETECTED', '???? Voice/Noise Detected detected by camera', '2026-03-04 22:36:38'),
(86, 14, 27, 'TAB_SWITCH', 'Window lost focus (1 times)', '2026-03-04 22:43:51'),
(87, 14, 27, 'NO_FACE', 'Camera appears blocked/covered', '2026-03-04 22:44:10'),
(88, 14, 27, 'TAB_SWITCH', 'Window lost focus (1 times)', '2026-03-04 22:44:32'),
(89, 14, 27, 'NO_FACE', 'Camera appears blocked/covered', '2026-03-04 22:44:32'),
(90, 14, 27, 'VOICE_DETECTED', '???? Voice/Noise Detected detected by camera', '2026-03-04 22:44:32'),
(91, 14, 28, 'VOICE_DETECTED', '???? Voice/Noise Detected detected by camera', '2026-03-04 22:54:36'),
(92, 14, 28, 'VOICE_DETECTED', '???? Voice/Noise Detected detected by camera', '2026-03-04 22:54:36'),
(93, 14, 28, 'VOICE_DETECTED', '???? Voice/Noise Detected detected by camera', '2026-03-04 22:54:36'),
(94, 14, 29, 'NO_FACE', 'Camera appears blocked/covered', '2026-03-04 22:55:33'),
(95, 14, 29, 'TAB_SWITCH', 'Window lost focus (1 times)', '2026-03-04 22:55:55'),
(96, 14, 29, 'NO_FACE', 'Camera appears blocked/covered', '2026-03-04 22:55:57'),
(97, 14, 29, 'NO_FACE', '???? No Face Detected detected by camera', '2026-03-04 22:55:57'),
(98, 14, 29, 'TAB_SWITCH', 'Window lost focus (1 times)', '2026-03-04 22:55:57'),
(99, 14, 30, 'TAB_SWITCH', 'Window lost focus (1 times)', '2026-03-04 23:22:57'),
(100, 14, 30, 'VOICE_DETECTED', 'Continuous human voice detected from microphone', '2026-03-04 23:23:19'),
(101, 14, 30, 'NO_FACE', 'Camera appears blocked/covered', '2026-03-04 23:23:33'),
(102, 14, 30, 'TAB_SWITCH', 'Window lost focus (1 times)', '2026-03-04 23:23:36'),
(103, 14, 30, 'VOICE_DETECTED', 'Continuous human voice detected from microphone', '2026-03-04 23:23:36'),
(104, 14, 30, 'NO_FACE', 'Camera appears blocked/covered', '2026-03-04 23:23:36');

--
-- Indexes for dumped tables
--

--
-- Indexes for table `exam_results`
--
ALTER TABLE `exam_results`
  ADD PRIMARY KEY (`ResultID`),
  ADD KEY `idx_exam_results_student` (`StudentID`),
  ADD KEY `idx_exam_results_session` (`SessionID`);

--
-- Indexes for table `exam_sessions`
--
ALTER TABLE `exam_sessions`
  ADD PRIMARY KEY (`SessionID`),
  ADD KEY `idx_exam_sessions_student` (`StudentID`);

--
-- Indexes for table `profiles`
--
ALTER TABLE `profiles`
  ADD PRIMARY KEY (`id`),
  ADD UNIQUE KEY `unique_student_profile` (`student_id`);

--
-- Indexes for table `students`
--
ALTER TABLE `students`
  ADD PRIMARY KEY (`ID`),
  ADD UNIQUE KEY `Email` (`Email`);

--
-- Indexes for table `violations`
--
ALTER TABLE `violations`
  ADD PRIMARY KEY (`ViolationID`),
  ADD KEY `idx_violations_student` (`StudentID`),
  ADD KEY `idx_violations_session` (`SessionID`);

--
-- AUTO_INCREMENT for dumped tables
--

--
-- AUTO_INCREMENT for table `exam_results`
--
ALTER TABLE `exam_results`
  MODIFY `ResultID` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=30;

--
-- AUTO_INCREMENT for table `exam_sessions`
--
ALTER TABLE `exam_sessions`
  MODIFY `SessionID` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=31;

--
-- AUTO_INCREMENT for table `profiles`
--
ALTER TABLE `profiles`
  MODIFY `id` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=4;

--
-- AUTO_INCREMENT for table `students`
--
ALTER TABLE `students`
  MODIFY `ID` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=15;

--
-- AUTO_INCREMENT for table `violations`
--
ALTER TABLE `violations`
  MODIFY `ViolationID` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=105;

--
-- Constraints for dumped tables
--

--
-- Constraints for table `exam_results`
--
ALTER TABLE `exam_results`
  ADD CONSTRAINT `exam_results_ibfk_1` FOREIGN KEY (`StudentID`) REFERENCES `students` (`ID`) ON DELETE CASCADE,
  ADD CONSTRAINT `exam_results_ibfk_2` FOREIGN KEY (`SessionID`) REFERENCES `exam_sessions` (`SessionID`) ON DELETE CASCADE;

--
-- Constraints for table `exam_sessions`
--
ALTER TABLE `exam_sessions`
  ADD CONSTRAINT `exam_sessions_ibfk_1` FOREIGN KEY (`StudentID`) REFERENCES `students` (`ID`) ON DELETE CASCADE;

--
-- Constraints for table `profiles`
--
ALTER TABLE `profiles`
  ADD CONSTRAINT `profiles_ibfk_1` FOREIGN KEY (`student_id`) REFERENCES `students` (`ID`) ON DELETE CASCADE;

--
-- Constraints for table `violations`
--
ALTER TABLE `violations`
  ADD CONSTRAINT `violations_ibfk_1` FOREIGN KEY (`StudentID`) REFERENCES `students` (`ID`) ON DELETE CASCADE,
  ADD CONSTRAINT `violations_ibfk_2` FOREIGN KEY (`SessionID`) REFERENCES `exam_sessions` (`SessionID`) ON DELETE CASCADE;
COMMIT;

/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
