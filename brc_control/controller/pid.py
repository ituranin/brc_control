class PIDController:
    def __init__(self, kp, ki, kd, setpoint = 0):

        self.kp = kp  # Proportional gain
        self.ki = ki  # Integral gain
        self.kd = kd  # Derivative gain
        self.setpoint = setpoint  # Desired value

        # Internal variables to store PID computations
        self._previous_error = 0.0
        self._integral = 0.0
        self.p_term = 0.0
        self.i_term = 0.0
        self.d_term = 0.0

    def update(self, current_value):
        # Calculate error
        error = self.setpoint - current_value
        
        # Proportional term
        p_term = self.kp * error
        
        # Integral term
        self._integral += error
        i_term = self.ki * self._integral
        
        # Derivative term

        derivative = (error - self._previous_error)
        d_term = self.kd * derivative
       
        # Update previous error for next iteration
        self._previous_error = error
        
        # Control output
        control_output = p_term + i_term + d_term
        self.p_term = p_term
        self.i_term = i_term
        self.d_term = d_term
        return control_output

    def reset_controller(self):
        self._previous_error = 0.0
        self._integral = 0.0

    def set_setpoint(self, setpoint):
        self.setpoint = setpoint

    def get_gains(self):
        return self.kp, self.ki, self.kd 
    
    def set_gains(self, kp=None, ki=None, kd=None):
        if kp is not None:
            self.kp = kp
        if ki is not None:
            self.ki = ki
        if kd is not None:
            self.kd = kd
