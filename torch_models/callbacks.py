class EarlyStopping:
    def __init__(self, monitor='val_loss', min_delta=1e-10, patience=10, verbose=1):
        self.monitor = monitor  # 모니터링할 지표 (검증 손실 등)
        self.min_delta = min_delta  # 개선을 감지하기 위한 최소한의 변화
        self.patience = patience  # 몇 번의 에폭 동안 개선이 없으면 훈련을 중지할 것인지
        self.verbose = verbose  # 출력 메시지를 표시할지 여부
        self.counter = 0  # patience를 세는 카운터
        self.best_score = None  # 최적의 지표 값
        self.early_stop = False  # 조기 중지 여부

    def __call__(self, current_score):
        if self.best_score is None:
            self.best_score = current_score
        elif current_score < self.best_score - self.min_delta:
            if self.verbose:
                print(f'Validation {self.monitor} decreased by {self.min_delta}. Saving model ...')
            self.best_score = current_score
            self.counter = 0
        else:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                if self.verbose:
                    print('Early stopping')
                self.early_stop = True