-- MySQL Script generated by MySQL Workbench
-- Thu Apr  1 10:15:39 2021
-- Model: New Model    Version: 1.0
-- MySQL Workbench Forward Engineering

SET @OLD_UNIQUE_CHECKS=@@UNIQUE_CHECKS, UNIQUE_CHECKS=0;
SET @OLD_FOREIGN_KEY_CHECKS=@@FOREIGN_KEY_CHECKS, FOREIGN_KEY_CHECKS=0;
SET @OLD_SQL_MODE=@@SQL_MODE, SQL_MODE='ONLY_FULL_GROUP_BY,STRICT_TRANS_TABLES,NO_ZERO_IN_DATE,NO_ZERO_DATE,ERROR_FOR_DIVISION_BY_ZERO,NO_ENGINE_SUBSTITUTION';

-- -----------------------------------------------------
-- Schema Yeyland_Wutani
-- -----------------------------------------------------

-- -----------------------------------------------------
-- Schema Yeyland_Wutani
-- -----------------------------------------------------
CREATE SCHEMA IF NOT EXISTS `Yeyland_Wutani` DEFAULT CHARACTER SET utf8mb4 ;
USE `Yeyland_Wutani` ;

-- -----------------------------------------------------
-- Table `Yeyland_Wutani`.`Department`
-- -----------------------------------------------------
DROP TABLE IF EXISTS `Yeyland_Wutani`.`Department` ;

CREATE TABLE IF NOT EXISTS `Yeyland_Wutani`.`Department` (
  `depnum` CHAR(5) NOT NULL,
  `depname` VARCHAR(45) NOT NULL,
  PRIMARY KEY (`depnum`),
  UNIQUE INDEX `depnum_UNIQUE` (`depnum` ASC) VISIBLE)
ENGINE = InnoDB;

INSERT INTO department (depnum, depname)
VALUES ('dep01', 'Accounting'),
('dep02', 'Production'),
('dep03', 'Developmemt'),
('dep04', 'Research'),
('dep05', 'Education'),
('dep06', 'Management'),
('dep07', 'IT');
-- -----------------------------------------------------
-- Table `Yeyland_Wutani`.`Position`
-- -----------------------------------------------------
DROP TABLE IF EXISTS `Yeyland_Wutani`.`Position` ;

CREATE TABLE IF NOT EXISTS `Yeyland_Wutani`.`Position` (
  `positioncode` CHAR(5) NOT NULL,
  `pdescription` MEDIUMTEXT NULL,
  PRIMARY KEY (`positioncode`),
  UNIQUE INDEX `positioncode_UNIQUE` (`positioncode` ASC) VISIBLE)
ENGINE = InnoDB;

INSERT INTO position (positioncode, pdescription)
VALUES ('po001','manager'),
('po002', 'president'), 
('po003', 'data entry specialist'), 
('po004', 'professional'),
('po005', 'accountant');
-- -----------------------------------------------------
-- Table `Yeyland_Wutani`.`Location`
-- -----------------------------------------------------
DROP TABLE IF EXISTS `Yeyland_Wutani`.`Location` ;

CREATE TABLE IF NOT EXISTS `Yeyland_Wutani`.`Location` (
  `locationcode` INT NOT NULL,
  `location` VARCHAR(45) NOT NULL,
  PRIMARY KEY (`locationcode`),
  UNIQUE INDEX `locationcode_UNIQUE` (`locationcode` ASC) VISIBLE)
ENGINE = InnoDB;

INSERT INTO location (locationcode, location)
VALUES (2408,'Ås'),
(1200, 'Oslo'), 
(1516, 'Moss'), 
(1110, 'Oslo'),
(2310, 'Ås'),
(1501, 'Moss');

-- -----------------------------------------------------
-- Table `Yeyland_Wutani`.`Adresses`
-- -----------------------------------------------------
DROP TABLE IF EXISTS `Yeyland_Wutani`.`Adresses` ;

CREATE TABLE IF NOT EXISTS `Yeyland_Wutani`.`Adresses` (
  `adressecode` CHAR(5) NOT NULL,
  `adresse` VARCHAR(45) NOT NULL,
  `postalcode` INT NOT NULL,
  PRIMARY KEY (`adressecode`),
  UNIQUE INDEX `adressecode_UNIQUE` (`adressecode` ASC) VISIBLE,
  INDEX `fk_postalcode_locationcode_idx` (`postalcode` ASC) VISIBLE,
  CONSTRAINT `postalcode`
    FOREIGN KEY (`postalcode`)
    REFERENCES `Yeyland_Wutani`.`Location` (`locationcode`)
    ON DELETE NO ACTION
    ON UPDATE NO ACTION)
ENGINE = InnoDB;

INSERT INTO adresses (adressecode, adresse, postalcode)
VALUES ('a0001','Nabogata 8', 1516),
('a0002', 'Havlandet 98', 1110),
('a0003', 'Biblioteket 15', '1200'),
('a0004', 'Dueberget 2A', '2408' ),
('a0005', 'Duegata 13A', '1501'),
('a0006', 'Hoppebakken 2A', '2310'),
('a0007', 'Storgata 4', '1200'),
('a0008', 'Skoleveien 4', '2408'),
('a0009', 'Storgata 4', '1200'),
('a0010','Skibakken 2', '1516'),
('a0011','Akerveien 54', '1110'),
('a0012','Lysaker 7', '2310'),
('a0013','Mosseveien 1C', '1501'),
('a0014','Lysaker 7', '2310'),
('a0015','Mossgata 14', '1516');


-- -----------------------------------------------------
-- Table `Yeyland_Wutani`.`Employee`
-- -----------------------------------------------------
DROP TABLE IF EXISTS `Yeyland_Wutani`.`Employee` ;

CREATE TABLE IF NOT EXISTS `Yeyland_Wutani`.`Employee` (
  `eID` CHAR(5) NOT NULL COMMENT 'Employee ID',
  `lname` VARCHAR(45) NOT NULL,
  `fname` VARCHAR(45) NOT NULL,
  `phone` INT NOT NULL,
  `depnum` CHAR(5) NOT NULL,
  `positioncode` CHAR(5) NOT NULL,
  `adresse_code` CHAR(5) NOT NULL,
  PRIMARY KEY (`eID`),
  INDEX `fk_depnum_idx` (`depnum` ASC) VISIBLE,
  INDEX `fk_postioncode_idx` (`positioncode` ASC) VISIBLE,
  UNIQUE INDEX `eID_UNIQUE` (`eID` ASC) VISIBLE,
  INDEX `fk_adresse_code_idx` (`adresse_code` ASC) VISIBLE,
  CONSTRAINT `depnum`
    FOREIGN KEY (`depnum`)
    REFERENCES `Yeyland_Wutani`.`Department` (`depnum`)
    ON DELETE NO ACTION
    ON UPDATE NO ACTION,
  CONSTRAINT `postioncode`
    FOREIGN KEY (`positioncode`)
    REFERENCES `Yeyland_Wutani`.`Position` (`positioncode`)
    ON DELETE NO ACTION
    ON UPDATE NO ACTION,
  CONSTRAINT `adresse_code`
    FOREIGN KEY (`adresse_code`)
    REFERENCES `Yeyland_Wutani`.`Adresses` (`adressecode`)
    ON DELETE NO ACTION
    ON UPDATE NO ACTION)
ENGINE = InnoDB;

INSERT INTO employee (eID, lname, fname, phone, depnum, positioncode, adresse_code)
VALUES ('emp1', 'Henriksen', 'Ola', '12345678', 'dep06', 'po001', 'a0001'),
('emp2', 'Beeblebrox', 'Zaphod', '56788765', 'dep03', 'po002', 'a0002'),
('emp3', 'Anderson', 'Thomas', '34566543', 'dep07', 'po003', 'a0003'),
('emp4', 'Reno', 'Leon', '98766789', 'dep04', 'po004', 'a0004'),
('emp5', 'Durden', 'Tyler', '76544567',  'dep06', 'po001', 'a0005'),
('emp6', 'Larsen', 'Hanne', '23455432',  'dep01', 'po005', 'a0006');

-- -----------------------------------------------------
-- Table `Yeyland_Wutani`.`Course`
-- -----------------------------------------------------
DROP TABLE IF EXISTS `Yeyland_Wutani`.`Course` ;

CREATE TABLE IF NOT EXISTS `Yeyland_Wutani`.`Course` (
  `courseID` CHAR(5) NOT NULL,
  `coursename` VARCHAR(45) NOT NULL,
  `cdescription` LONGTEXT NULL,
  `lecturer` VARCHAR(45) NOT NULL,
  PRIMARY KEY (`courseID`),
  UNIQUE INDEX `courseID_UNIQUE` (`courseID` ASC) VISIBLE)
ENGINE = InnoDB;

INSERT INTO course (courseID, coursename, cdescription, lecturer)
VALUES ('co001', 'Artisan soap making 101', 'A beginner course in soap making.', 'David Williams'),
('co002', 'Advanced anger management', 'An advanced course in anger management for employees.', 'May Olsen'),
('co003', 'Artisan tea making 101', 'A beginner course in alternative methods of tea making.', 'Ann Clarke'),
('co004', 'Spoon bending for beginners', 'An advanced course in spoon bending.', 'Alice Hill'),
('co005', 'Artisan soup making 101', 'A beginner course in soup making.', 'John Adams'),
('co006', 'Building better worlds', 'An advanced course in management.', 'Michelle Michael'),
('co007', 'Alien ecology', 'A beginner course in alien ecology.', 'Rose Harris');

-- -----------------------------------------------------
-- Table `Yeyland_Wutani`.`Course_Schedule`
-- -----------------------------------------------------
DROP TABLE IF EXISTS `Yeyland_Wutani`.`Course_Schedule` ;

CREATE TABLE IF NOT EXISTS `Yeyland_Wutani`.`Course_Schedule` (
  `scheduleID` CHAR(5) NOT NULL,
  `courseID` CHAR(5) NOT NULL,
  `schedule_datetime` DATETIME NOT NULL,
  `adr_code` CHAR(5) NOT NULL,
  PRIMARY KEY (`scheduleID`),
  INDEX `fk_courseID_idx` (`courseID` ASC) VISIBLE,
  UNIQUE INDEX `scheduleID_UNIQUE` (`scheduleID` ASC) VISIBLE,
  INDEX `fk_adr_code_idx` (`adr_code` ASC) VISIBLE,
  CONSTRAINT `courseID`
    FOREIGN KEY (`courseID`)
    REFERENCES `Yeyland_Wutani`.`Course` (`courseID`)
    ON DELETE NO ACTION
    ON UPDATE NO ACTION,
  CONSTRAINT `adr_code`
    FOREIGN KEY (`adr_code`)
    REFERENCES `Yeyland_Wutani`.`Adresses` (`adressecode`)
    ON DELETE NO ACTION
    ON UPDATE NO ACTION)
ENGINE = InnoDB;

INSERT INTO course_schedule (scheduleID, courseID, schedule_datetime, adr_code )
VALUES ('s0001','co001', '2017-06-03 17:00:00', 'a0007'),
('s0002','co002', '2017-10-14 12:30:00', 'a0008'),
('s0003','co001', '2017-07-18 14:30:00', 'a0009'),
('s0004','co003', '2017-06-03 15:00:00', 'a0010'),
('s0005','co004', '2017-03-22 17:00:00', 'a0011'),
('s0006','co004', '2017-04-01 18:00:00', 'a0011'),
('s0007','co004', '2017-04-08 20:00:00', 'a0011'),
('s0008','co004', '2017-04-22 09:00:00', 'a0011'),
('s0009','co005', '2017-01-02 09:30:00', 'a0012'),
('s0010','co006', '2017-01-01 10:30:00', 'a0013'),
('s0011','co006', '2017-12-12 12:00:00', 'a0013'),
('s0012','co007', '2017-02-11 20:30:00', 'a0014');

-- -----------------------------------------------------
-- Table `Yeyland_Wutani`.`Course_Attendance`
-- -----------------------------------------------------
DROP TABLE IF EXISTS `Yeyland_Wutani`.`Course_Attendance` ;

CREATE TABLE IF NOT EXISTS `Yeyland_Wutani`.`Course_Attendance` (
  `eID` CHAR(5) NOT NULL,
  `scheduleID` CHAR(5) NOT NULL,
  PRIMARY KEY (`eID`, `scheduleID`),
  INDEX `fk_scheduleID_idx` (`scheduleID` ASC) VISIBLE,
  INDEX `fk_eID_idx` (`eID` ASC) VISIBLE,
  CONSTRAINT `fk_eID`
    FOREIGN KEY (`eID`)
    REFERENCES `Yeyland_Wutani`.`Employee` (`eID`)
    ON DELETE NO ACTION
    ON UPDATE NO ACTION,
  CONSTRAINT `fk_scheduleID`
    FOREIGN KEY (`scheduleID`)
    REFERENCES `Yeyland_Wutani`.`Course_Schedule` (`scheduleID`)
    ON DELETE NO ACTION
    ON UPDATE NO ACTION)
ENGINE = InnoDB;

INSERT INTO course_attendance(eID, scheduleID)
VALUES ('emp5', 's0001'),
('emp1', 's0001'),
('emp6', 's0001'),
('emp5', 's0002'),
('emp1', 's0002'),
('emp6', 's0002'),
('emp5', 's0003'),
('emp2', 's0004'),
('emp2', 's0005'),
('emp3', 's0006'),
('emp3', 's0007'),
('emp3', 's0008'),
('emp1', 's0009');

SET SQL_MODE=@OLD_SQL_MODE;
SET FOREIGN_KEY_CHECKS=@OLD_FOREIGN_KEY_CHECKS;
SET UNIQUE_CHECKS=@OLD_UNIQUE_CHECKS;