-- 한줄 주석 (--공백 으로 시작) + (#공백 도 가능)
/* block 주석 */

/*
Mysql Workbench
  - font 바꾸기 - edit/preferences -> font and color
  
  - SQL문 실행 - control + enter (cmd + enter)
*/


/**********************************************************************************************************
사용자 계정 생성
- create user 'username'@'host' identified by 'password'
  - usernamer과 host를 따로 작은 따옴표로 묶어준다. -> 작은 따옴표만 문자열 나타냄 
  - host
    - localhost : 로컬 접속 계정
    - % : 원격 접속 계정

**********************************************************************************************************/
-- local 접속 계정

create user 'playdata'@'localhost' identified by '1111';

-- 원격 접속 계정

create user 'playdata'@'%' identified by '1111';

-- 등록된 사용자계정 조회

select user, host from mysql.user;


/********************************************************
 계정에 권한 부여
- GRANT 부여할 권한 ON 데이터베이스.테이블 TO 계정@host
- 데이터베이스와 테이블을 `*` 로 지정하면 모든 DB와 테이블에 적용된다.
- 주요 권한 목록
  - all privileges: 모든권한
  - 테이블의 데이터 관리: select, insert, update, delete
  - DB 객체 관리: create, drop, alter
  - 사용자관리: create user, drop user, grant option
********************************************************/

grant all privileges on *.* to 'playdata'@'localhost';
grant all privileges on *.* to 'playdata'@'%';

-- user 권한 조회
show grants for 'playdata'@'localhost';
show grants for 'playdata'@'%';

/*************************************************
Database 생성

CREATE DATABASE testdb;
*************************************************/

create database testdb;

-- DB 확인
show databases;

# create table testdb.member
-- 특정 DB를 사용 -> database이름을 넣을 곳에 생략하면 설정한 DB를 사용한다
-- use database 이름
use testdb;
-- create table member

/***********************************************************************************************************
테이블 생성
create table 테이블 이름 (컬럼명  데이터타입  [제약조건])

테이블 삭제
drop table [if exists] 삭제할테이블이름

------------------------------------------------------------------------------------------------------------
테이블: 회원 (member)
속성
id:        varchar(10)    primary key
password:  varchar(10)    not null (필수)
name:      varchar(30)    not null
point:     int            nullable
email:     varchar(100)   unique key
gender:    char(1)        not null, check key - 'm', 'f' 만 값으로 가진다.
age:       int            check key - 양수만 값으로 가진다.
join_date: timestamp      not null, 기본값-값 저장시 일시
***********************************************************************************************************/
-- 테이블 생성
create table member(
	id varchar(10)  primary key,
    password varchar(10)  not null,
    name varchar(30)  not null,
    point int,
    email varchar(100)  unique key,
    gender char(1) not null check(gender in ('m', 'f')),
    age int check(age > 0),
    join_date timestamp not null  default current_timestamp -- insert 시점의 일시를 저장
);

-- 테이블 조회
show tables;

-- 테이블 상세정보를 조회(칼럼)
describe member;
desc member;

-- 테이블 삭제
drop table if exists member;


/* *********************************************************************
INSERT 문 - 행 추가
구문
 - 한행추가 :
   - INSERT INTO 테이블명 (컬럼 [, 컬럼]) VALUES (값 [, 값[])
     - 모든 컬럼에 값을 넣을 경우 컬럼 지정구문은 생략 할 수 있다.
     - SQL은 기본적으로 한 행(한개의 데이터)씩 테이블에 추가한다.

************************************************************************ */
insert into member (id, password, name, point, email, gender, age, join_date)
values ('id-1', '1111', '홍길동', 1000, 'h@s.com', 'm', 20, '2025-10-14 17:25:20');

insert into member values ('id-2', '1111', '홍길동', 1000, 'h2@s.com', 'm', 20, '2025-10-14 17:25:20');

-- join_date는 default 값(sql 실행 일수)가 insert됨
insert into member (id, password, name, gender) values ('id-3', '2222', '이순신', 'm');

insert into member  (id, password, name, gender, age) values ('id-4', '2222', '유관순', 'f', 20);

insert into member  (id, password, name, gender, age, join_date) values ('id-5', '2222', '유관순', 'f', 20, '2000-10-10');
-- timestamp 에 날짜만 insert -> 시간은 00:00:00 로 출력

insert into member (id, password, name, gender, age) values ('id-6', '2222', '이순신', 'm', null);
-- null(결측치) --> password는 불가능 : not null 제약조건 때문에

select * from member;


