from pathlib import Path
from typing import List, Union
import torch
import torch.nn

class Dynamics:
    """Base building dynamics model."""

    def __init__(self):
        pass

    def reset(self):
        pass

class LSTMDynamics(Dynamics, torch.nn.Module):
    """LSTM building dynamics model that predicts indoor temperature based on partial cooling/heating load and other weather variables.
    
    Parameters
    ----------
    filepath: Union[Path, str]
        Path to model state dictionary.
    input_observation_names: List[str]
        List of maximum values used for input observation min-max normalization.
    input_normalization_minimum: List[float]
        List of minumum values used for input observation min-max normalization.
    input_normalization_maximum: List[float]
        List of maximum values used for input observation min-max normalization.
    hidden_size: int
        The number of neurons in hidden layer.
    num_layers: int
        Number of hidden layers.
    lookback: int
        Number of samples used for prediction.
    input_size: int, optional
        Number of variables used for prediction. This may not equal `input_observation_names`
        e.g. cooling and heating demand may be included in `input_observation_names` but only
        one of two may be used for the actual prediction depending on building needs.
        The default is to set set `input_size` to the length of `input_observation_names`.
    dropout: float, default: 0.0
        Probability of excluding input and recurrent connections to LSTM units from activation 
        and weight updates while training a network. This has the effect of reducing overfitting 
        and improving model performance.
    """

    def __init__(
            self, filepath: Union[Path, str], input_observation_names: List[str], input_normalization_minimum: List[float], 
            input_normalization_maximum: List[float], hidden_size: int, num_layers: int, lookback: int, input_size: int = None,
            dropout: float = None
    ):
        Dynamics.__init__(self)
        torch.nn.Module.__init__(self)
        assert len(input_observation_names) == len(input_normalization_minimum) == len(input_normalization_maximum),\
            'input_observation_names, input_normalization_minimum and input_normalization_maximum must have the same length.'
        self.input_observation_names = input_observation_names
        self.input_normalization_minimum = input_normalization_minimum
        self.input_normalization_maximum = input_normalization_maximum
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lookback = lookback
        self.filepath = filepath
        self.l_lstm = self.set_lstm()
        self.l_linear = self.set_linear()
        self._hidden_state = None
        self._model_input = None
        self.dropout = torch.nn.Dropout(dropout if dropout is not None else 0.0)

    @property
    def input_size(self) -> int:
        return self.__input_size
    
    @input_size.setter
    def input_size(self, value: int):
        self.__input_size = len(self.input_observation_names) if value is None else value
    
    def set_lstm(self) -> torch.nn.LSTM:
        """Initialize LSTM model."""

        return torch.nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
        )

    def set_linear(self) -> torch.nn.Linear:
        """Initialize linear transformer."""

        return torch.nn.Linear(
            in_features=self.hidden_size,
            out_features=1
        )
    
    def forward(self, x, h):
        """Predict indoor dry bulb temperature."""

        lstm_out, h = self.l_lstm(x, h)
        lstm_out = self.dropout(lstm_out)
        out = lstm_out[:, -1, :]
        out_linear_transf = self.l_linear(out)
        return out_linear_transf, h

    def init_hidden(self, batch_size: int):
        """Initialize hidden states."""

        hidden_state = torch.zeros(self.num_layers, batch_size, self.hidden_size)
        cell_state = torch.zeros(self.num_layers, batch_size, self.hidden_size)
        hidden = (hidden_state, cell_state)
        
        return hidden

    def reset(self):
        """Loads dynamic model state dict, and initializes hidden states and model input."""

        super().reset()

        try:
            self.load_state_dict(torch.load(self.filepath)['model_state_dict'])
        
        except RuntimeError:
            self.load_state_dict(torch.load(self.filepath, map_location=torch.device('cpu'))['model_state_dict'])
        
        except:
            self.load_state_dict(torch.load(self.filepath))

        self._hidden_state = self.init_hidden(1)
        self._model_input = [[None]*(self.lookback + 1) for _ in self.input_observation_names]
    # LSTMDYNAMICS-STEP
    def step(self, obs, u_cool=None):
        """Predict next indoor dry bulb temperature (°C) using the LSTM surrogate.

        - Uses schema-provided input_observation_names and min/max for min-max normalization.
        - Maintains an internal rolling window via self._model_input.
        - Optional control link: if 'cooling_demand' is an input and u_cool is provided,
          we inject action dependence by using partial cooling: cooling_demand := cooling_demand * clamp(u_cool,0,1).
        """

        def _get(k, default=None):
            if obs is None:
                return default
            try:
                if hasattr(obs, "get"):
                    return obs.get(k, default)
                return obs[k]
            except Exception:
                return default

        # Ensure model buffers exist
        if self._hidden_state is None or self._model_input is None:
            self.reset()

        # Clamp action
        u = None
        if u_cool is not None:
            try:
                u = float(u_cool)
            except Exception:
                u = 0.0
            if u < 0.0:
                u = 0.0
            elif u > 1.0:
                u = 1.0

        # Update rolling input buffer (each feature has a queue length lookback+1)
        for i, name in enumerate(self.input_observation_names):
            v = _get(name, None)

            # Inject action dependence via partial cooling load if applicable
            if name == "cooling_demand" and (u is not None) and (v is not None):
                try:
                    v = float(v) * (1.0 - u)
                except Exception:
                    pass

            # Replace missing values safely
            if v is None:
                prev = self._model_input[i][-1]
                v = 0.0 if prev is None else float(prev)
            else:
                try:
                    v = float(v)
                except Exception:
                    v = 0.0

            # shift left + append
            self._model_input[i] = self._model_input[i][1:] + [v]

        # Build normalized tensor x: (1, lookback, input_size)
        # Use last `lookback` samples from each feature queue
        X_feats = []
        for i in range(self.input_size):
            vals = self._model_input[i][-self.lookback:]
            # replace any None (can happen at episode start)
            vals = [0.0 if v is None else float(v) for v in vals]

            mn = float(self.input_normalization_minimum[i])
            mx = float(self.input_normalization_maximum[i])
            denom = (mx - mn) if (mx - mn) != 0.0 else 1.0

            X_feats.append([max(0.0, min(1.0, (v - mn) / denom)) for v in vals])

        # transpose to (lookback, features)
        X_time = list(zip(*X_feats))

        import torch
        x = torch.tensor([X_time], dtype=torch.float32)  # (1, lookback, features)

        with torch.no_grad():
            y, h = self.forward(x, self._hidden_state)
            self._hidden_state = h

        y_val = float(y.view(-1)[0])

        # Denormalize output back to °C using indoor temp min/max (must exist in these models)
        if "indoor_dry_bulb_temperature" in self.input_observation_names:
            j = self.input_observation_names.index("indoor_dry_bulb_temperature")
            mn = float(self.input_normalization_minimum[j])
            mx = float(self.input_normalization_maximum[j])
            y_val = y_val * (mx - mn) + mn

        return float(y_val)



    def terminate(self):
        return
class RCDynamics(Dynamics):
    """Realistic minimal 1R1C thermal dynamics with HVAC cooling control.

    State:
        Tin (°C)

    Inputs:
        Tout (°C) from observations (e.g., 'outdoor_dry_bulb_temperature')
        u_cool in [0,1] (cooling action fraction)

    Parameters:
        R_K_per_kW: thermal resistance (K/kW)
        C_kWh_per_K: thermal capacitance (kWh/K)
        hvac_max_kW: max cooling power removed from zone (kW)
        dt_seconds: timestep size (s)

    Update (Euler):
        Q_env = (Tout - Tin)/R        [kW]
        Q_cool = u * hvac_max_kW      [kW]  (removes heat)
        Tin_next = Tin + (dt_h/C) * (Q_env - Q_cool)

    Notes:
      - Backward compatible: if Tout not found in obs, uses Tout = Tin (no passive drift).
      - Safety clamps are applied if provided.
    """

    def __init__(
        self,
        dt_seconds: float,
        R_K_per_kW: float = 2.0,
        C_kWh_per_K: float = 3.0,
        hvac_max_kW: float = 10.0,
        initial_indoor_temperature: float = 22.0,
        outdoor_temperature_observation_name: str = "outdoor_dry_bulb_temperature",
        min_indoor_temperature: float = 10.0,
        max_indoor_temperature: float = 35.0,
        **kwargs
    ):
        super().__init__()
        self.dt_seconds = float(dt_seconds)
        self.R_K_per_kW = float(R_K_per_kW)
        self.C_kWh_per_K = float(C_kWh_per_K)
        self.hvac_max_kW = float(hvac_max_kW)
        self.T0 = float(initial_indoor_temperature)
        self.outdoor_temperature_observation_name = str(outdoor_temperature_observation_name)
        self.min_indoor_temperature = float(min_indoor_temperature)
        self.max_indoor_temperature = float(max_indoor_temperature)
        self.T = self.T0  # current indoor temperature

    def reset(self):
        super().reset()
        self.T = self.T0

    def step(self, obs, u_cool):
        # Clamp action
        try:
            u = float(u_cool)
        except Exception:
            u = 0.0
        if u < 0.0:
            u = 0.0
        elif u > 1.0:
            u = 1.0

        Tin = float(self.T)

        # Read Tout from obs; if missing, default to Tin (backward compatible)
        Tout = Tin
        key = self.outdoor_temperature_observation_name
        try:
            if obs is not None:
                if hasattr(obs, "get"):
                    v = obs.get(key, None)
                else:
                    v = obs[key]
                if v is not None:
                    Tout = float(v)
        except Exception:
            Tout = Tin

        dt_h = self.dt_seconds / 3600.0

        # Heat flow from outdoors (kW)
        Q_env = (Tout - Tin) / self.R_K_per_kW

        # Cooling removes heat (kW)
        Q_cool = u * self.hvac_max_kW

        # Temperature update
        Tin_next = Tin + (dt_h / self.C_kWh_per_K) * (Q_env - Q_cool)

        # Safety clamps
        if Tin_next < self.min_indoor_temperature:
            Tin_next = self.min_indoor_temperature
        if Tin_next > self.max_indoor_temperature:
            Tin_next = self.max_indoor_temperature

        self.T = float(Tin_next)
        return float(Tin_next)
