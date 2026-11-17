"""The keyboard as a driver's hands and feet: keys that ramp the wheel and the pedals instead of jumping.

A key is on or off, but a driver's hand turns the wheel over a moment and lets it come back to centre. Holding
an arrow moves the wheel toward full lock at STEER_RISE; letting go brings it back to centre at STEER_RETURN,
which is faster, as a real wheel springs back. Pressing the other arrow while the wheel is turned first brings
it back to centre at the return rate, then turns it the other way at the rise rate. The pedals ramp up the same
way and drop to nothing at once when the key is let go. car.step narrows the lock with speed on its own.
"""

from caiman.car import Controls

STEER_RISE = 0.30  # seconds from centre to full lock
STEER_RETURN = 0.18  # seconds from full lock back to centre
PEDAL_RISE = 0.20  # seconds from a released pedal to a pedal on the floor


def _toward(value: float, target: float, rate: float, dt: float) -> float:
    """value moved toward target by at most rate * dt."""
    if value < target:
        return min(value + rate * dt, target)
    return max(value - rate * dt, target)


class Driver:
    """Keeps the wheel and the pedals between frames and turns each frame's keys into Controls."""

    def __init__(self):
        self.steer = 0.0
        self.throttle = 0.0
        self.brake = 0.0

    def update(self, left: bool, right: bool, up: bool, down: bool, handbrake: bool, dt: float) -> Controls:
        """The controls after dt seconds with these keys held."""
        target = float(right) - float(left)
        if target == 0 or self.steer * target < 0:
            self.steer = _toward(self.steer, 0.0, 1 / STEER_RETURN, dt)
        else:
            self.steer = _toward(self.steer, target, 1 / STEER_RISE, dt)
        self.throttle = min(self.throttle + dt / PEDAL_RISE, 1.0) if up else 0.0
        self.brake = min(self.brake + dt / PEDAL_RISE, 1.0) if down else 0.0
        return Controls(throttle=self.throttle, brake=self.brake, steer=self.steer, handbrake=handbrake)
