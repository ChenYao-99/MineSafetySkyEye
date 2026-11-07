"""Bounded recovery of an exited owned child;never terminate an occupied service."""
from collections import deque
import threading,time


class DetectorSupervisor:
    def __init__(self,get_child,restart,port_active,*,interval=2,max_attempts=3,window_seconds=300,clock=time.monotonic):
        self.get_child=get_child;self.restart=restart;self.port_active=port_active
        self.interval=interval;self.max_attempts=max_attempts;self.window_seconds=window_seconds;self.clock=clock
        self.attempts=deque();self.stop_event=threading.Event();self.state='not_started';self.total_attempts=0;self.thread=None

    def tick(self):
        child=self.get_child()
        if child is None:self.state='no_owned_child';return
        if child.poll() is None:self.state='child_running';return
        if self.port_active():self.state='port_occupied';return
        now=self.clock()
        while self.attempts and now-self.attempts[0]>=self.window_seconds:self.attempts.popleft()
        if len(self.attempts)>=self.max_attempts:self.state='retry_budget_wait';return
        self.attempts.append(now);self.total_attempts+=1;self.state='recovering'
        print('[detector] exited child;starting bounded recovery attempt',self.total_attempts,flush=True)
        try:
            self.restart();current=self.get_child()
            self.state='child_running' if current is not None and current.poll() is None else 'restart_failed'
        except Exception as error:
            self.state='restart_failed';print('[detector] recovery failed:',type(error).__name__,flush=True)

    def start(self):
        if self.thread and self.thread.is_alive():return
        def run():
            while not self.stop_event.wait(self.interval):
                try:self.tick()
                except Exception as error:
                    self.state='monitor_failed';print('[detector] monitor check failed:',type(error).__name__,flush=True)
        self.thread=threading.Thread(target=run,name='kat-detector-supervisor',daemon=True);self.thread.start()

    def snapshot(self):
        return {'state':self.state,'recovery_attempts':self.total_attempts,'max_attempts_per_window':self.max_attempts,
            'retry_window_seconds':self.window_seconds,'owned_child_present':self.get_child() is not None,
            'child_running_does_not_imply_healthy':True}
