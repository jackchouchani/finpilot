import React, { useState } from 'react';
import { TextField, Button, Box } from '@mui/material';
import axios from 'axios';

function Register({ onRegisterSuccess }) {
    const [username, setUsername] = useState('');
    const [password, setPassword] = useState('');

    const handleSubmit = async (e) => {
        e.preventDefault();
        try {
            await axios.post(process.env.REACT_APP_API_URL + '/register', { username, password });
            alert('Registration successful! Please log in.');
            onRegisterSuccess();
        } catch (error) {
            alert('Registration failed. Please try again.');
            console.error("Registration error:", error);
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
                Register
            </Button>
        </Box>
    );
}

export default Register;
