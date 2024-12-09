- 문제 정의 및 목표
Diffusion model 기반 생성형 인공지능이 만들어낸 가짜 이미지와 진짜 이미지를 분류

- 대회 평가 지표 및 개발 환경
데이터: Diffusion model로 생성된 이미지와 진짜 이미지
평가지표: F1 Score
참여인원: 5인
개발 기간: 2023.05 ~ 2024.07

- 문제 분석 및 해결
Frequency domain 분석:
생성형 인공지능은 주파수 도메인에서 특정 흔적을 남기며, 이는 GAN과 Diffusion model 의 종류에 따라 다른 형태로 나타난다.
형태는 서로 다르지만 고주파 성분에서 이러한 흔적이 두드러진다는 공통된 특징을 가지고 있다.

고주파 성분 강조:
학습 및 추론 시 데이터에 Pillow 라이브러리에서 제공하는 High Pass Filter 인 EDGE_ENHNACE 를 적용하여 고주파 성분을 강조

모델 선정 및 학습:
Swin Transformer 를 최종 모델로 선정하여 학습 및 추론 진행

- 결과 및 성과
최종 F1 Score: 0.862
등수: 22등 / 상위 4%
수상: 최초 제출과 마지막 제출의 점수 차이가 가장 큰 팀으로 선정되어 “특별상: 역전의 명탐정” 수상
![image](https://github.com/user-attachments/assets/f938b52c-122e-4485-ad7f-8cf6c854cd66)
