use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};

#[pyfunction]
fn hash_observation_rust(obs: Bound<'_, PyDict>) -> PyResult<u128> {
    // Get health (0-2)
    let health: u128 = obs.get_item("agent_health")?
        .and_then(|v| v.extract().ok())
        .unwrap_or(0);

    // Get window
    let window_item = obs.get_item("window")?
        .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>("Missing window"))?;
    let window = window_item.downcast::<PyDict>()?;

    // Build cell values in order
    let mut cell_values: Vec<u128> = Vec::with_capacity(25);

    for dr in -2..=2 {
        for dc in -2..=2 {
            // Create tuple key (dr, dc) - must match Python exactly
            let key = {
                let py = window.py();
                PyTuple::new_bound(py, &[dr, dc])
            };

            let cell_opt = window.get_item(&key)?;

            if let Some(cell_any) = cell_opt {
                let cell = cell_any.downcast::<PyDict>()?;

                // Check in_bounds
                let in_bounds: bool = cell.get_item("in_bounds")?
                    .and_then(|v| v.extract().ok())
                    .unwrap_or(false);

                if !in_bounds {
                    cell_values.push(8);
                    continue;
                }

                // Determine cell value by priority
                let is_wall: bool = cell.get_item("is_wall")?
                    .and_then(|v| v.extract().ok())
                    .unwrap_or(false);

                if is_wall {
                    cell_values.push(1);
                    continue;
                }

                let is_goal: bool = cell.get_item("is_goal")?
                    .and_then(|v| v.extract().ok())
                    .unwrap_or(false);

                if is_goal {
                    cell_values.push(7);
                    continue;
                }

                let has_portal: bool = cell.get_item("has_portal")?
                    .and_then(|v| v.extract().ok())
                    .unwrap_or(false);

                if has_portal {
                    let portal_color: Option<String> = cell.get_item("portal_color")?
                        .and_then(|v| v.extract().ok());

                    match portal_color.as_deref() {
                        Some("red") => cell_values.push(4),
                        Some("blue") => cell_values.push(5),
                        Some("green") => cell_values.push(6),
                        _ => cell_values.push(0),
                    }
                    continue;
                }

                let has_minotaur: bool = cell.get_item("has_minotaur")?
                    .and_then(|v| v.extract().ok())
                    .unwrap_or(false);

                if has_minotaur {
                    cell_values.push(3);
                    continue;
                }

                let is_trap: bool = cell.get_item("is_trap")?
                    .and_then(|v| v.extract().ok())
                    .unwrap_or(false);

                if is_trap {
                    cell_values.push(2);
                    continue;
                }

                // Empty cell
                cell_values.push(0);
            } else {
                cell_values.push(8);
            }
        }
    }

    // Pack into base-9 integer
    let mut window_hash: u128 = 0;
    let mut base: u128 = 1;

    for &value in &cell_values {
        window_hash = window_hash.wrapping_add(value.wrapping_mul(base));
        base = base.wrapping_mul(9);
    }

    // Get minotaur ID
    let minotaur_id: u128 = obs.get_item("minotaur_in_cell")?
        .and_then(|v| {
            if v.is_none() {
                Some(0)
            } else {
                v.extract::<String>().ok()
                    .and_then(|s| s.chars().last())
                    .and_then(|c| c.to_digit(10))
                    .map(|d| d as u128)
            }
        })
        .unwrap_or(0);

    // Calculate final state ID - MUST MATCH PYTHON EXACTLY
    // Python: WINDOW_SPACE = 9 ** 25
    let window_space: u128 = 717_897_987_691_852_588_770_249;  // This is 9^25 in Python's calculation
    let minotaur_space: u128 = window_space;
    let health_space: u128 = minotaur_space * 6;

    let state_id = health.wrapping_mul(health_space)
        .wrapping_add(minotaur_id.wrapping_mul(minotaur_space))
        .wrapping_add(window_hash);

    Ok(state_id)
}

#[pymodule]
fn state_hash_rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(hash_observation_rust, m)?)?;
    Ok(())
}