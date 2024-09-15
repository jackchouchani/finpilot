import React, { useState, useEffect } from 'react';
import { TextField, Button, Select, MenuItem, Chip, Box, FormControl, InputLabel } from '@mui/material';
import axios from 'axios';

function Settings({ onClearChat }) {
    const [settings, setSettings] = useState({
        default_portfolio_value: 100000,
        risk_profile: 'moderate',
        preferred_sectors: []
    });

    useEffect(() => {
        fetchSettings();
    }, []);

    const fetchSettings = async () => {
        try {
            const token = localStorage.getItem('token');
            const response = await axios.get(process.env.REACT_APP_API_URL + '/settings', {
                headers: { 'Authorization': `Bearer ${token}` }
            });
            setSettings({
                ...response.data,
                preferred_sectors: Array.isArray(response.data.preferred_sectors)
                    ? response.data.preferred_sectors
                    : []
            });
        } catch (error) {
            console.error("Error fetching settings:", error);
        }
    };

    const handleChange = (event) => {
        setSettings({ ...settings, [event.target.name]: event.target.value });
    };

    const handleSectorChange = (event) => {
        setSettings({ ...settings, preferred_sectors: event.target.value });
    };

    const handleSettingsChange = (setting, value) => {
        setSettings(prevSettings => ({ ...prevSettings, [setting]: value }));
    };

    const clearChat = async () => {
        try {
            await axios.post(process.env.REACT_APP_API_URL + '/clear_chat');
            alert('Chat history cleared successfully');
            onClearChat(); // Appel de la fonction pour mettre à jour l'état dans App.js
        } catch (error) {
            console.error('Error clearing chat history:', error);
            alert('Error clearing chat history. Please try again.');
        }
    };

    const saveSettings = async () => {
        try {
            const token = localStorage.getItem('token');
            const response = await axios.post(process.env.REACT_APP_API_URL + '/settings', settings, {
                headers: { 'Authorization': `Bearer ${token}` }
            });
            if (response.data.message === "Settings updated successfully") {
                alert('Settings saved successfully');
            } else {
                throw new Error('Unexpected response');
            }
        } catch (error) {
            console.error('Error saving settings:', error);
            alert('Error saving settings. Please try again.');
        }
    };

    return (
        <Box>
            <TextField
                label="Default Portfolio Value"
                value={settings.default_portfolio_value}
                onChange={(e) => handleSettingsChange('default_portfolio_value', e.target.value)}
                type="number"
                fullWidth
                margin="normal"
            />
            <FormControl fullWidth margin="normal">
                <InputLabel>Risk Profile</InputLabel>
                <Select
                    value={settings.risk_profile}
                    onChange={(e) => handleSettingsChange('risk_profile', e.target.value)}
                >
                    <MenuItem value="low">Low</MenuItem>
                    <MenuItem value="moderate">Moderate</MenuItem>
                    <MenuItem value="high">High</MenuItem>
                </Select>
            </FormControl>
            <Select
                label="Preferred Sectors"
                name="preferred_sectors"
                multiple
                value={settings.preferred_sectors}
                onChange={handleSectorChange}
                renderValue={(selected) => (
                    <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5 }}>
                        {selected.map((value) => (
                            <Chip key={value} label={value} />
                        ))}
                    </Box>
                )}
                fullWidth
                margin="normal"
            >
                <MenuItem value="Technology">Technology</MenuItem>
                <MenuItem value="Healthcare">Healthcare</MenuItem>
                <MenuItem value="Finance">Finance</MenuItem>
                <MenuItem value="Energy">Energy</MenuItem>
            </Select>
            <Button
                onClick={clearChat}
                variant="contained"
                color="secondary"
                style={{ marginTop: '20px' }}
            >
                Clear Chat History
            </Button>
            <Button onClick={saveSettings} variant="contained" color="primary">
                Save Settings
            </Button>
        </Box>
    );
}

export default Settings;