#!/usr/bin/env pwsh
# Fix admin role for user dced8a9f-5f89-4d44-a146-9f7070793749

$ErrorActionPreference = 'Stop'

Write-Host '[INFO] Fixing admin role for user...' -ForegroundColor Cyan

# Load .env file
$envPath = Join-Path $PSScriptRoot '..' '.env'
if (Test-Path $envPath) {
    Get-Content $envPath | ForEach-Object {
        if ($_ -match '^([^#][^=]+)=(.*)$') {
            $key = $matches[1].Trim()
            $value = $matches[2].Trim().Trim('"')
            [Environment]::SetEnvironmentVariable($key, $value, 'Process')
        }
    }
    Write-Host '[OK] Loaded .env file' -ForegroundColor Green
} else {
    Write-Host '[ERROR] .env file not found at: $envPath' -ForegroundColor Red
    exit 1
}

$supabaseUrl = $env:VITE_SUPABASE_URL
$supabaseKey = $env:VITE_SUPABASE_ANON_KEY
$userId = 'dced8a9f-5f89-4d44-a146-9f7070793749'

if (-not $supabaseUrl -or -not $supabaseKey) {
    Write-Host '[ERROR] Missing Supabase credentials in .env' -ForegroundColor Red
    exit 1
}

Write-Host "[INFO] Supabase URL: $supabaseUrl" -ForegroundColor Cyan
Write-Host "[INFO] User ID: $userId" -ForegroundColor Cyan

# Check if user exists in users_roles_settings
Write-Host '[INFO] Checking if user exists in users_roles_settings...' -ForegroundColor Cyan
$checkUrl = "$supabaseUrl/rest/v1/users_roles_settings?user_id=eq.$userId&select=user_id,role_code"
$headers = @{
    'apikey' = $supabaseKey
    'Authorization' = "Bearer $supabaseKey"
    'Content-Type' = 'application/json'
}

try {
    $response = Invoke-RestMethod -Uri $checkUrl -Method Get -Headers $headers
    
    if ($response -and $response.Count -gt 0) {
        Write-Host "[INFO] User exists with role: $($response[0].role_code)" -ForegroundColor Yellow
        
        # Update role to admin
        Write-Host '[INFO] Updating role to admin...' -ForegroundColor Cyan
        $updateUrl = "$supabaseUrl/rest/v1/users_roles_settings?user_id=eq.$userId"
        $updateBody = @{
            role_code = 'admin'
        } | ConvertTo-Json
        
        $headers['Prefer'] = 'return=representation'
        $updateResponse = Invoke-RestMethod -Uri $updateUrl -Method Patch -Headers $headers -Body $updateBody
        
        Write-Host '[OK] Role updated to admin successfully!' -ForegroundColor Green
        Write-Host "Updated user: $($updateResponse[0].user_id)" -ForegroundColor Green
        Write-Host "New role: $($updateResponse[0].role_code)" -ForegroundColor Green
    } else {
        Write-Host '[INFO] User does not exist in users_roles_settings, inserting...' -ForegroundColor Yellow
        
        # Insert new row with admin role
        $insertUrl = "$supabaseUrl/rest/v1/users_roles_settings"
        $insertBody = @{
            user_id = $userId
            role_code = 'admin'
            user_created_at = (Get-Date).ToUniversalTime().ToString('yyyy-MM-ddTHH:mm:ss.fffZ')
        } | ConvertTo-Json
        
        $headers['Prefer'] = 'return=representation'
        $insertResponse = Invoke-RestMethod -Uri $insertUrl -Method Post -Headers $headers -Body $insertBody
        
        Write-Host '[OK] User inserted with admin role successfully!' -ForegroundColor Green
        Write-Host "Inserted user: $($insertResponse[0].user_id)" -ForegroundColor Green
        Write-Host "Role: $($insertResponse[0].role_code)" -ForegroundColor Green
    }
    
    Write-Host ''
    Write-Host '[SUCCESS] Admin role fix completed!' -ForegroundColor Green
    Write-Host '[INFO] Please refresh the dashboard to see changes' -ForegroundColor Cyan
    
} catch {
    Write-Host "[ERROR] Failed to fix admin role: $_" -ForegroundColor Red
    Write-Host "Response: $($_.Exception.Response)" -ForegroundColor Red
    exit 1
}
