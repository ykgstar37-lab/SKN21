# polls/views.py
from django.http import HttpResponse
from django.shortcuts import render
from datetime import datetime
from .models import Question, Choice

# Create your views here.

# 설문 welcome page view
#  요청 -> 인삿말 화면을 응답.
def welcome(request):
    print("welcome 실행")
    # 요청 처리
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    # 응답 화면 생성 -> template 호출(template이 사용할 값을 전달-now)
    response = render( # template 호출 -결과-> HttpResponse로 반환
        request, # HttpRequest
        "polls/welcome.html", # 호출할 template의 경로
        {"now":now} # template에 전달할 값들. name-value 전달.
                    # Context Value 라고 한다.
    )
    print(type(response)) # server를 실행한 터미널에 출력.
    return response

def welcome_old(request):
    print("welcome 실행")
    # 요청 처리
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    # 처리결과 페이지 생성 -> html을 str로 구현.
    res_html = f"""<!doctype html>
<html>
    <head>
        <title>설문 main</title>
    </head>
    <body>
        <h1>Welcome</h1>
        <p>저희 설문 페이지를 방문해 주셔서 감사합니다.</p>
        현재 시간: {now}
    </body>
</html>
"""
    res = HttpResponse(res_html)
    return res 


# 설문(질문) 목록 조회
## 전체 question들을 조회해서 목록 html을 반환.
## 요청 url: polls/list
## view함수: list
## template: polls/list.html
def list(request):
    # 1. DB에서 질문 목록 조회 -> Model 사용
    question_list = Question.objects.all().order_by("-pub_date")
    # 2. 응답 페이지를 생성(template 사용) -> 반환
    return render(
        request, 
        "polls/list.html", 
        {"question_list":question_list}
    )


# 개별 설문을 할 수 있는 페이지(설문폼)로 이동
## 질문 id를 path parameter로 받아서
##      그 질문과 보기를 조화해서 화면에 출력 - 설문 입력 폼
### path parameter: http://ip:port/path1/path2/path3/전달값1/전달값2
### request(요청) parameter:
#          http://ip:port/path1/path2/path3?name=전달값1&name=전달값2

# 요청URL: /polls/vote_form/질문ID
# view함수: vote_form
# template: polls/vote_form.html
def vote_form(request, question_id):
    # question_id 파라미터 - path parameter 값을 받을 변수
    # view합수의 두번째 파라미터부터는 path parameter를 받을 변수들
    ## 파라미터 변수명은 uris.py에 등록한 변수명으로 선언하면 된다

    # 1. DB에서 question_id로 질문을 조회
    try:
        question = Question.objects.get(pk=question_id)
        # 응답 화면 요청
        render(
            request,
            "polls/vote_form.html",
            {"question":question}
        )

    except:
        # print(f"{question_id}의 질문이 없습니다")
        return render(
            request,
            "polls/error.html",
            {"error_message":f"{question_id}의 질문이 없습니다."}
        )