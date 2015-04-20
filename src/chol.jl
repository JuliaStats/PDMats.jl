# needs special attention to Cholesky, as the syntax & behavior changes in 0.4-pre

if VERSION >= v"0.4.0-dev+4370"
    error("Choleksy changes in 0.4.0-dev+4370 (PR #10862), we are still working to make it work with these changes.")
end
