import React, { useState } from 'react';
import { TextField, Button, Box } from '@mui/material';
import { login } from './Auth'

function Login({ onLogin }) {
    const [username, setUsername] = useState('');
    const [password, setPassword] = useState('');

    const handleSubmit = async (e) => {
        e.preventDefault();
        const success = await login(username, password);
        if (success) {
            onLogin();
        } else {
            alert('Login failed');
        }
    };

    return (
        <Box component="form" onSubmit={handleSubmit}>
            <TextField
                label="Username"
                value={username}
                onChange={(e) => setUsername(e.target.value)}
                fullWidth
                margin="normal"
            />
            <TextField
                label="Password"
                type="password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                fullWidth
                margin="normal"
            />
            <Button type="submit" variant="contained" color="primary">
                Login
            </Button>
        </Box>
    );
}

export default Login;